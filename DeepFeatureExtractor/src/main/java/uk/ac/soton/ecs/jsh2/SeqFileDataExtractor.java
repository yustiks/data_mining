package uk.ac.soton.ecs.jsh2;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.util.Map;
import java.util.Map.Entry;
import java.util.zip.ZipOutputStream;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.openimaj.hadoop.sequencefile.SequenceFileUtility;

/**
 * A concrete implementation of a {@link SequenceFileUtility} for
 * {@link SequenceFile}s with {@link Text} keys and {@link BytesWritable}
 * values.
 *
 * @author Jonathon Hare (jsh2@ecs.soton.ac.uk)
 */
class TextTextSequenceFileUtility extends SequenceFileUtility<Text, Text> {

	public TextTextSequenceFileUtility(String uriOrPath, boolean read) throws IOException {
		super(uriOrPath, read);
	}

	public TextTextSequenceFileUtility(String uriOrPath, CompressionType compressionType, Map<String, String> metadata)
			throws IOException
	{
		super(uriOrPath, compressionType, metadata);
	}

	public TextTextSequenceFileUtility(String uriOrPath, CompressionType compressionType) throws IOException {
		super(uriOrPath, compressionType);
	}

	public TextTextSequenceFileUtility(URI uri, boolean read) throws IOException {
		super(uri, read);
	}

	public TextTextSequenceFileUtility(URI uri, CompressionType compressionType, Map<String, String> metadata)
			throws IOException
	{
		super(uri, compressionType, metadata);
	}

	public TextTextSequenceFileUtility(URI uri, CompressionType compressionType) throws IOException {
		super(uri, compressionType);
	}

	@Override
	protected Text readFile(FileSystem fs, Path path) throws IOException {
		FSDataInputStream dis = null;
		ByteArrayOutputStream baos = null;

		try {
			dis = fs.open(path);
			baos = new ByteArrayOutputStream();

			IOUtils.copyBytes(dis, baos, config, false);

			final byte[] bytes = baos.toByteArray();
			return new Text(bytes);
		} finally {
			if (dis != null)
				try {
					dis.close();
				} catch (final IOException e) {
				}
			;
			if (baos != null)
				try {
					baos.close();
				} catch (final IOException e) {
				}
			;
		}
	}

	@Override
	protected void writeFile(FileSystem fs, Path path, Text value) throws IOException {
		FSDataOutputStream dos = null;

		try {
			dos = fs.create(path);
			final byte[] bytes = new byte[value.getLength()];
			System.arraycopy(value.getBytes(), 0, bytes, 0, bytes.length);
			dos.write(bytes);
		} finally {
			if (dos != null)
				try {
					dos.close();
				} catch (final IOException e) {
				}
			;
		}
	}

	@Override
	protected void printFile(Text value) throws IOException {
		System.out.write(value.getBytes());
	}

	@Override
	protected void writeZipData(ZipOutputStream zos, Text value) throws IOException {
		zos.write(value.getBytes(), 0, value.getLength());
	}
}

public class SeqFileDataExtractor extends Configured implements Tool {
	@Override
	public int run(String[] args) throws Exception {
		final String[] inputPathOrUri = new String[args.length - 1];
		for (int i = 1; i < args.length; i++)
			inputPathOrUri[i - 1] = args[i];

		final Path[] sequenceFiles = SequenceFileUtility.getFilePaths(inputPathOrUri, "part");

		for (final Path path : sequenceFiles) {
			System.out.println("Extracting from " + path.getName());

			final SequenceFileUtility<Text, Text> utility = new TextTextSequenceFileUtility(path.toUri(), true);

			for (final Entry<Text, Text> e : utility) {
				System.out.println(e.getKey() + ", " + e.getValue());
			}
		}

		return 0;
	}

	public static void main(String[] args) throws Exception {
		System.exit(ToolRunner.run(new SeqFileDataExtractor(), args));
	}
}
