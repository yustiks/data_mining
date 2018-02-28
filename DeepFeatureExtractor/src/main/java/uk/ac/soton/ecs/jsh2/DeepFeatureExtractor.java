package uk.ac.soton.ecs.jsh2;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Map;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.util.ModelSerializer;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openimaj.hadoop.sequencefile.SequenceFileUtility;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

/**
 * An attempt at a ResNet50-based feature extractor that can be deployed on
 * hadoop
 *
 * @author Jonathon Hare (jsh2@ecs.soton.ac.uk)
 *
 */
@SuppressWarnings("deprecation")
public class DeepFeatureExtractor extends Configured implements Tool {
	public static class Options {
		@Option(
				name = "--input",
				aliases = "-i",
				required = true,
				usage = "Input image sequence file directory.",
				metaVar = "STRING")
		String input;

		@Option(name = "--output", aliases = "-o", required = true, usage = "Output Sequence File", metaVar = "STRING")
		String output;

		@Option(name = "--rescale", required = false, usage = "Should the images be rescaled and cropped")
		boolean rescale = false;

		/**
		 * @return the input paths
		 * @throws IOException
		 */
		public Path[] getInputPaths() throws IOException {
			final Path[] sequenceFiles = SequenceFileUtility.getFilePaths(input, "part");
			return sequenceFiles;
		}

		/**
		 * @return the output path
		 */
		public Path getOutputPath() {
			return new Path(SequenceFileUtility.convertToURI(output).toString());
		}
	}

	public static class FeatureMap extends Mapper<Text, BytesWritable, Text, Text> {
		final static String RESCALE_KEY = "FeatureMap.RESCALE";
		final static int IMAGE_SIZE = 224;
		private ResizeProcessor rp;
		private ComputationGraph network;
		private VGG16ImagePreProcessor scaler;
		boolean rescale;

		static enum Counters {
			SUCCEEDED, FAILED
		}

		@Override
		protected void setup(Mapper<Text, BytesWritable, Text, Text>.Context context)
				throws IOException, InterruptedException
		{
			super.setup(context);

			rp = new ResizeProcessor(IMAGE_SIZE);

			network = ModelSerializer.restoreComputationGraph(
					DeepFeatureExtractor.class.getResourceAsStream("resnet50_dl4j_inference.zip"));

			scaler = new VGG16ImagePreProcessor();

			rescale = context.getConfiguration().getBoolean(RESCALE_KEY, false);
		}

		/**
		 * Convert an RGB {@link MBFImage} to a 4-dimensional {@link INDArray} tensor
		 * with dimensions {1,bands,height,width} and the bands stored in BGR format
		 *
		 * @param image
		 *            the image
		 * @return the tensor
		 */
		public static INDArray image2array(MBFImage image) {
			final int height = image.getHeight();
			final int width = image.getWidth();
			final int bands = image.numBands();

			// byte[] pixels = ((DataBufferByte)
			// image.getRaster().getDataBuffer()).getData();
			final int[] shape = new int[] { 1, bands, height, width };
			final INDArray ret = Nd4j.create(shape);

			for (int b = 0; b < bands; b++) {
				final int bprime = bands - b - 1;
				final float[][] pixels = image.bands.get(b).pixels;
				for (int j = 0; j < height; j++) {
					for (int i = 0; i < width; i++) {
						ret.putScalar(0, bprime, j, i, pixels[j][i] * 255);
					}
				}
			}

			return ret;
		}

		@Override
		protected void map(Text key, BytesWritable value,
				Mapper<Text, BytesWritable, Text, Text>.Context context)
				throws IOException, InterruptedException
		{
			try {
				// the hadoop framework gives us the image as raw bytes
				final ByteArrayInputStream bais = new ByteArrayInputStream(value.getBytes(), 0, value.getLength());

				// we load it into an image
				MBFImage image = ImageUtilities.readMBF(bais);
				// scale and crop
				if (rescale)
					image = image.processInplace(rp).extractCenter(IMAGE_SIZE, IMAGE_SIZE);

				// convert to a tensor and mean-center
				final INDArray input = image2array(image);
				scaler.preProcess(input);

				// then feed through the network. We'll cache __all__ activations/feature maps
				final Map<String, INDArray> activations = network.feedForward(input, false);

				// and because we have all of them we can write a few different vectors out:
				// the avg_pool layer - most likely to be useful for transfer learning
				context.write(new Text(key.toString() + "_avg_pool"), encode(activations.get("avg_pool")));

				// the final fully connected layer - this gives you the class probs over the
				// imagenet classes. Could be useful for learning...
				context.write(new Text(key.toString() + "fc1000"), encode(activations.get("fc1000")));

				// whilst we're here, we might as well decode the fc1000 layer into image top 5
				// imagenet classes (perhaps use these as 'tags' in a metadata-based model?
				context.write(new Text(key.toString() + "fc1000"),
						new Text(TrainedModels.VGG16.decodePredictions(activations.get("fc1000"))));

				context.getCounter(Counters.SUCCEEDED).increment(1L);
			} catch (final Throwable e) {
				System.gc(); // we can sometimes recover if we ran out of memory by flushing the GC... or we
								// might just crash and burn...
				context.getCounter(Counters.FAILED).increment(1L);
				e.printStackTrace();
			}
		}

		/**
		 * Encode a tensor as a comma separated string
		 *
		 * @param indArray
		 * @return
		 */
		private Text encode(INDArray indArray) {
			final INDArray flat = Nd4j.toFlattened(indArray);
			final int len = flat.shape()[1];
			String s = flat.getScalar(0) + "";
			for (int i = 1; i < len; i++) {
				s += ", " + flat.getScalar(i);
			}

			return new Text(s);
		}
	}

	@Override
	public int run(String[] args) throws Exception {
		final Options options = new Options();
		final CmdLineParser parser = new CmdLineParser(options);
		try {
			parser.parseArgument(args);
		} catch (final CmdLineException e) {
			System.err.println(e.getMessage());
			System.err
					.println(
							"Usage: yarn jar DeepFeatureExtractor.jar uk.ac.soton.ecs.jsh2.DeepFeatureExtractor [options...]");
			parser.printUsage(System.err);

			System.exit(1);
		}

		final Job job = new Job(this.getConf());
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);

		SequenceFileInputFormat.setInputPaths(job, options.getInputPaths());
		SequenceFileOutputFormat.setOutputPath(job, options.getOutputPath());
		SequenceFileOutputFormat.setCompressOutput(job, true);
		SequenceFileOutputFormat.setOutputCompressorClass(job, DefaultCodec.class);
		SequenceFileOutputFormat.setOutputCompressionType(job, CompressionType.BLOCK);

		job.setJarByClass(this.getClass());
		job.setMapperClass(FeatureMap.class);
		job.setNumReduceTasks(0); // no reducer is required!

		this.getConf().setBoolean(FeatureMap.RESCALE_KEY, options.rescale);

		long start, end;
		start = System.currentTimeMillis();
		job.waitForCompletion(true);
		end = System.currentTimeMillis();

		System.out.println("Took: " + (end - start) + "ms");

		return 0;
	}

	public static void main(String[] args) throws Exception {
		System.exit(ToolRunner.run(new DeepFeatureExtractor(), args));
	}
}
