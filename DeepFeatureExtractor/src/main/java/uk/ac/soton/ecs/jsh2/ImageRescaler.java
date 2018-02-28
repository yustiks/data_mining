package uk.ac.soton.ecs.jsh2;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

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
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.openimaj.hadoop.sequencefile.SequenceFileUtility;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

/**
 * A mapper that reduces and crops images
 *
 * @author Jonathon Hare (jsh2@ecs.soton.ac.uk)
 *
 */
public class ImageRescaler extends Configured implements Tool {
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

	/**
	 * Subclasses ByteArrayOutputStream to give access to the internal raw buffer.
	 */
	public static class MyByteArrayOutputStream extends ByteArrayOutputStream {
		public MyByteArrayOutputStream() {
			super();
		}

		public MyByteArrayOutputStream(int size) {
			super(size);
		}

		/**
		 * Returns the internal buffer of this ByteArrayOutputStream, without copying.
		 */
		public synchronized byte[] buf() {
			return this.buf;
		}
	}

	/**
	 * The Mapper implementation
	 *
	 * @author Jonathon Hare (jsh2@ecs.soton.ac.uk)
	 *
	 */
	public static class Rescaler extends Mapper<Text, BytesWritable, Text, BytesWritable> {
		final static int IMAGE_SIZE = 224;
		private ResizeProcessor rp;
		private MyByteArrayOutputStream baos;
		private BytesWritable buffer;

		@Override
		protected void setup(Mapper<Text, BytesWritable, Text, BytesWritable>.Context context)
				throws IOException, InterruptedException
		{
			super.setup(context);
			rp = new ResizeProcessor(IMAGE_SIZE);
			baos = new MyByteArrayOutputStream(2048);
			buffer = new BytesWritable();
		}

		static enum Counters {
			SUCCEEDED, FAILED
		}

		@Override
		protected void map(Text key, BytesWritable value,
				Mapper<Text, BytesWritable, Text, BytesWritable>.Context context)
				throws IOException, InterruptedException
		{
			try {
				// the hadoop framework gives us the image as raw bytes
				final ByteArrayInputStream bais = new ByteArrayInputStream(value.getBytes(), 0, value.getLength());

				// we load it into an image
				MBFImage image = ImageUtilities.readMBF(bais);
				// scale and crop
				image = image.processInplace(rp).extractCenter(IMAGE_SIZE, IMAGE_SIZE);

				baos.reset(); // reset buffer pointer to beginning so we can reuse it
				// save it to a buffer
				ImageUtilities.write(image, "jpeg", baos);
				// and write it back out
				buffer.set(baos.buf(), 0, baos.size());
				context.write(key, buffer);

				context.getCounter(Counters.SUCCEEDED).increment(1L);
			} catch (final Throwable e) {
				System.gc(); // we can sometimes recover if we ran out of memory by flushing the GC... or we
								// might just crash and burn...
				context.getCounter(Counters.FAILED).increment(1L);
				e.printStackTrace();
			}
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
							"Usage: yarn jar DeepFeatureExtractor.jar uk.ac.soton.ecs.jsh2.ImageRescaler [options...]");
			parser.printUsage(System.err);

			System.exit(1);
		}

		final Job job = new Job(this.getConf());
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(BytesWritable.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);

		SequenceFileInputFormat.setInputPaths(job, options.getInputPaths());
		SequenceFileOutputFormat.setOutputPath(job, options.getOutputPath());
		SequenceFileOutputFormat.setCompressOutput(job, true);
		SequenceFileOutputFormat.setOutputCompressorClass(job, DefaultCodec.class);
		SequenceFileOutputFormat.setOutputCompressionType(job, CompressionType.BLOCK);

		job.setJarByClass(this.getClass());
		job.setMapperClass(Rescaler.class);
		job.setNumReduceTasks(0); // no reducer is required!

		long start, end;
		start = System.currentTimeMillis();
		job.waitForCompletion(true);
		end = System.currentTimeMillis();

		System.out.println("Took: " + (end - start) + "ms");

		return 0;
	}

	public static void main(String[] args) throws Exception {
		System.exit(ToolRunner.run(new ImageRescaler(), args));
	}
}
