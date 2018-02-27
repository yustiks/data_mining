package uk.ac.soton.ecs.jsh2;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;
import org.nd4j.linalg.factory.Nd4j;

import lombok.extern.slf4j.Slf4j;

/**
 * This is the VGG16ImagePreProcessor from dl4j, but with a fix applied to
 * correct the ordering of the means to actually be BGR rather than the RGB
 * order in the distributed version!!!
 *
 * @author Jon
 */
@Slf4j
public class VGG16ImagePreProcessor implements DataNormalization {
	private static final long serialVersionUID = 1L;

	public static final INDArray VGG_MEAN_OFFSET_BGR = Nd4j.create(new double[] { 103.939, 116.779, 123.68 });

	/**
	 * Fit a dataset (only compute based on the statistics from this dataset0
	 *
	 * @param dataSet
	 *            the dataset to compute on
	 */
	@Override
	public void fit(DataSet dataSet) {

	}

	/**
	 * Iterates over a dataset accumulating statistics for normalization
	 *
	 * @param iterator
	 *            the iterator to use for collecting statistics.
	 */
	@Override
	public void fit(DataSetIterator iterator) {

	}

	@Override
	public void preProcess(DataSet toPreProcess) {
		final INDArray features = toPreProcess.getFeatures();
		this.preProcess(features);
	}

	public void preProcess(INDArray features) {
		Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(features.dup(), VGG_MEAN_OFFSET_BGR, features, 1));
	}

	/**
	 * Transform the data
	 *
	 * @param toPreProcess
	 *            the dataset to transform
	 */
	@Override
	public void transform(DataSet toPreProcess) {
		this.preProcess(toPreProcess);
	}

	@Override
	public void transform(INDArray features) {
		this.preProcess(features);
	}

	@Override
	public void transform(INDArray features, INDArray featuresMask) {
		transform(features);
	}

	@Override
	public void transformLabel(INDArray label) {
		// No op
	}

	@Override
	public void transformLabel(INDArray labels, INDArray labelsMask) {
		transformLabel(labels);
	}

	@Override
	public void revert(DataSet toRevert) {
		revertFeatures(toRevert.getFeatures());
	}

	@Override
	public NormalizerType getType() {
		return NormalizerType.IMAGE_VGG16;
	}

	@Override
	public void revertFeatures(INDArray features) {
		Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(features.dup(), VGG_MEAN_OFFSET_BGR, features, 1));
	}

	@Override
	public void revertFeatures(INDArray features, INDArray featuresMask) {
		revertFeatures(features);
	}

	@Override
	public void revertLabels(INDArray labels) {
		// No op
	}

	@Override
	public void revertLabels(INDArray labels, INDArray labelsMask) {
		revertLabels(labels);
	}

	@Override
	public void fitLabel(boolean fitLabels) {

	}

	@Override
	public boolean isFitLabel() {
		return false;
	}
}
