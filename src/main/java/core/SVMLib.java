package core;

import com.google.common.base.Preconditions;
import libsvm.*;

import java.io.IOException;
import java.util.Vector;

/**
 *
 * Created by edwardlol on 16/8/15.
 */
public class SVMLib {
    //~ Static fields and initializer ------------------------------------------

    private static SVMLib instance = null;

    public static boolean DEBUG = false;

    //~ Instance fields --------------------------------------------------------

    public LibConfig config  = LibConfig.getInstance();

    Data trainingData;
    Data testData;
    public svm_parameter svm_param = this.config.getDefaultParam();

    //~ Constructors -----------------------------------------------------------

    private SVMLib() {}

    //~ Methods ----------------------------------------------------------------

    /**
     * get the only instance of this class
     * @return the only instance of this class
     */
    public static SVMLib getInstance() {
        if (instance == null) {
            instance = new SVMLib();
        }
        return instance;
    }

    public void initDataFromFile() {
        this.trainingData = new Data().readDataFromCSVFile(this.config.properties.getProperty("trainingData"));
    }

    /**
     * train the data sets using the given parameter
     * @param data training data sets
     * @param param the svm parameter to train the data
     * @return a trained model, can be used to validate training data
     */
    @org.jetbrains.annotations.Nullable
    public svm_model train(Data data, svm_parameter param) {
        Preconditions.checkNotNull(data);
        Preconditions.checkNotNull(param);

        long startTime = System.currentTimeMillis();
        /* set svm problem */
        svm_problem problem = new svm_problem();
        problem.l = data.getSampleNum();
        problem.x = new svm_node[problem.l][];
        problem.y = new double[problem.l];

        for(int i = 0; i < problem.l; i++) {
            problem.x[i] = data.getDataSet("scaled").get(i);
            problem.y[i] = data.getLabels().get(i);
        }
        /* train svm model */
        String errorMsg = svm.svm_check_parameter(problem, param);
        if (errorMsg == null) {
            svm_model model = svm.svm_train(problem, param);
            if (DEBUG) {
                try {
                    String modelFile = this.config.properties.getProperty("modelFile");
                    svm.svm_save_model(modelFile, model);
                } catch (IOException e) {
                    e.printStackTrace();
                } finally {
                    System.out.println("Train finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
                }
            }
            return model;
        } else {
            System.out.println(errorMsg);
            return null;
        }
    }

    /**
     *
     * @param sample
     * @param model
     * @return
     */
    @SuppressWarnings("unused")
    public double predict(double[] sample, svm_model model) {
        svm_node[] svm_sample = new svm_node[sample.length];
        for (int i = 0; i < sample.length; i++) {
            svm_node sample_feature = new svm_node();
            sample_feature.index = i + 1;
            sample_feature.value = sample[i];
        }
        return svm.svm_predict(model, svm_sample);
    }

    /**
     * do cross validation on given dataset and given parameter
     * @param data data
     * @param param param
     * @param fold_n the number of folds
     * @return the 'loss' of the prediction
     */
    private double crossValidation(Data data, svm_parameter param, int fold_n) {
        double totalDiff = 0.0d;
        for (int i = 0; i < fold_n; i++) {
            Vector<svm_node[]> trainSet = new Vector<>();
            Vector<svm_node[]> validSet = new Vector<>();
            Vector<Double> trainLabels = new Vector<>();
            Vector<Double> validLabels = new Vector<>();

            int vsLen = data.getSampleNum() / fold_n;
            int vsStart = i * vsLen;
            int vsEnd = (i + 1) * vsLen;

            for (int j = 0; j < vsStart; j++) {
                trainSet.add(data.getDataSet("scaled").get(j));
                trainLabels.add(data.getLabels().get(j));
            }
            for (int j = vsStart; j < vsEnd; j++) {
                validSet.add(data.getDataSet("scaled").get(j));
                validLabels.add(data.getLabels().get(j));
            }
            for (int j = vsEnd; j < data.getSampleNum(); j++) {
                trainSet.add(data.getDataSet("scaled").get(j));
                trainLabels.add(data.getLabels().get(j));
            }

            Data trainData = new Data(trainSet, trainLabels);
            svm_model model = train(trainData, param);
            if (model != null) {
                double diff = 0.0d;
                for (int j = 0; j < validSet.size(); j++) {
                    svm_node[] sample = validSet.get(i);
                    double real_label = validLabels.get(i);
                    double predict_label = svm.svm_predict(model, sample);
                    diff += Math.pow((predict_label - real_label), 2);
                }
                totalDiff += diff;
            }
        }
        return totalDiff / data.getSampleNum();
    }

    /**
     * use grid search to optimize svm_parameter
     * @param param
     * @param data the dataset to do grid search
     * @return the optimized svm_parameter
     */
    @SuppressWarnings("unused")
    public svm_parameter updateParam(svm_parameter param, Data data) {
        // suppress training outputs
        svm_print_interface print_func = this.config.svm_print_null;
        svm.svm_set_print_string_function(print_func);

        double bestC = 1.0d;
        double bestG = 1.0 / data.getSampleNum();
        double smallestDiff = Double.MAX_VALUE;

        for (int power_of_c = -8; power_of_c < 8; power_of_c += 1) {
            param.C = Math.pow(2, power_of_c);

            // check if default g gives better result
            param.gamma = 1.0 / data.getSampleNum();
            double diff = crossValidation(data, param, 10);
            if ((diff < smallestDiff)) {
                smallestDiff = diff;
                bestC = param.C;
                bestG = param.gamma;
                System.out.println("best c: " + bestC + "; best g: " + bestG + "; diff: " + diff);
            }

            for (int power_of_g = -8; power_of_g < 8; power_of_g += 1) {
                param.gamma = Math.pow(2, power_of_g);
                diff = crossValidation(data, param, 10);

                if ((diff < smallestDiff)) {
                    smallestDiff = diff;
                    bestC = param.C;
                    bestG = param.gamma;
                    System.out.println("best c: " + bestC + "; best g: " + bestG + "; diff: " + diff);
                }
            }
        }
        param.C = bestC;
        param.gamma = bestG;
        System.out.println("best C: " + param.C + "; best gamma: " + param.gamma + "; best diff: " + smallestDiff);
        return param;
    }

}

// End SVMLib.java
