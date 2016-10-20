package someone_else;

import com.google.common.collect.Maps;
import core.Data;
import core.LibConfig;
import core.SVMLib;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import org.junit.Test;

import java.io.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import java.util.Map;
import java.util.Vector;

/**
 *
 * Created by edwardlol on 16/8/13.
 */
public class RegressionDemo {
    //~ Static fields and initializer ------------------------------------------

    private static final DateTimeFormatter dateTimeFormatter
            = DateTimeFormatter.ofPattern("yyyy-MM-dd", Locale.CHINA);

    //~ Test methods -----------------------------------------------------------

    @Test
    public void regression1() {
        SVMLib svmLib = SVMLib.getInstance().setType(LibConfig.Type.REGRESSION);

        Data trainData = new Data();
        Data testData = new Data();
        /* read data */
        trainData.readDataFromCSVFile("./datasets/svr/train.csv");
        testData.readDataFromCSVFile("./datasets/svr/test.csv");
        /* record data */
        trainData.recordData("./results/svr/train", "original");
        testData.recordData("./results/svr/test", "original");

        svm_parameter param = svmLib.svm_param;
        param.gamma = 1.0 / trainData.getSampleNum();

        // uncomment this line to do cross validation and utilize the svm_param
        // caution: cost a lot of time
//        param = SVMLib.updateParam(param, trainData);

        svm_model model = svmLib.train(trainData, param);
        regressionResult(model, testData);
    }

    /**
     * regression of weather data and defect count
     */
    @Test
    public void regression2() {
        SVMLib svmLib = SVMLib.getInstance();

        Map<LocalDate, String> defcnt = readPart1("./datasets/svr/defcnt.csv");
        Map<LocalDate, Integer[]> weather = readPart2("./datasets/svr/weather.csv");

        Vector<svm_node[]> samples = new Vector<>();
        Vector<Double> labels = new Vector<>();

        for (Map.Entry<LocalDate, String> entry : defcnt.entrySet()) {
            LocalDate dateTime = entry.getKey();
            if (weather.containsKey(dateTime)) {
                svm_node[] sample = new svm_node[weather.get(dateTime).length];
                for (int i = 0; i < weather.get(dateTime).length; i++) {
                    sample[i] = new svm_node();
                    sample[i].index = i + 1;
                    sample[i].value = weather.get(dateTime)[i];
                }
                samples.add(sample);
                labels.add(Double.parseDouble(entry.getValue()));
            }
        }

        double trainingRatio = 0.9d;
        int trainingIndex = (int)Math.round(samples.size() * trainingRatio);

        Vector<svm_node[]> trainSamples = new Vector<>(samples.subList(0, trainingIndex));
        Vector<Double> trainLabels = new Vector<>(labels.subList(0, trainingIndex));
        Data trainData = new Data(trainSamples, trainLabels);

        Vector<svm_node[]> testSamples = new Vector<>(samples.subList(trainingIndex, samples.size()));
        Vector<Double> testLabels = new Vector<>(labels.subList(trainingIndex, samples.size()));
        Data testData = new Data(testSamples, testLabels);

        svm_parameter param = svmLib.svm_param;
        param.gamma = 1.0 / trainData.getSampleNum();

        // uncomment this line to do cross validation and utilize the svm_param
        // caution: cost a lot of time
//        param = SVMLib.updateParam(param, trainData);

        svm_model model = svmLib.train(trainData, param);
        regressionResult(model, testData);
    }

    //~ Helper methods ---------------------------------------------------------

    /**
     * read defect data and make a map
     * consists of date and defect count
     * @param file defect data
     * @return a map consists of date and defect count
     */
    private Map<LocalDate, String> readPart1(String file) {
        Map<LocalDate, String> defcnt = Maps.newHashMap();
        try (FileReader fr = new FileReader(file);
             BufferedReader br = new BufferedReader(fr)) {

            String line = br.readLine(); // first line is shit
            line = br.readLine();
            while (line != null) {
                String[] contents = line.split(",");
                LocalDate date = LocalDate.parse(contents[0], dateTimeFormatter);
                String cnt = contents[1];

                defcnt.put(date, cnt);
                line = br.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return defcnt;
    }

    /**
     * read weather data and make a map
     * consists of date and weather data
     * @param file weather data
     * @return a map consists of date and weather data
     */
    private Map<LocalDate, Integer[]> readPart2(String file) {
        Map<LocalDate, Integer[]> weather = Maps.newHashMap();
        try (FileReader fr = new FileReader(file);
             BufferedReader br = new BufferedReader(fr);) {

            String line = br.readLine();
            while (line != null) {
                String[] contents = line.split(",");
                int monthValue = Integer.parseInt(contents[1]);
                String month = String.format("%02d", monthValue);
                int dayValue = Integer.parseInt(contents[2]);
                String day = String.format("%02d", dayValue);

                LocalDate date = LocalDate.parse(contents[0] + "-" + month + "-" + day, dateTimeFormatter);
                Integer[] wthFeature = new Integer[4];
                for (int i = 0; i < wthFeature.length; i++) {
                    wthFeature[i] = Integer.parseInt(contents[i + 3]);
                }
                weather.put(date, wthFeature);
                line = br.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return weather;
    }

    /**
     *
     * @param model
     * @param data
     * @return
     */
    private static double regressionResult(svm_model model, Data data) {
        long startTime = System.currentTimeMillis();
        double diff = 0;
        int totalCnt = 0, goodCnt = 0;
        Vector<svm_node[]> set = data.getDataSet("original");
        Vector<Double> labels = data.getLabels();
        try (FileWriter fw = new FileWriter("./results/svr/result.txt");
             BufferedWriter bw = new BufferedWriter(fw)) {

            for (int i = 0; i < data.getSampleNum(); i++) {
                svm_node[] sample = set.get(i);
                double realLabel = labels.get(i);
                double predictLabel = svm.svm_predict(model, sample);
                bw.write("predict label: " + predictLabel + "; real label: " + realLabel + "; ");
                for (int j = 0; j < data.getFeatureNum(); j ++) {
                    bw.write(sample[j].index + ":" + sample[j].value + " ");
                }
                bw.write("\n");
                bw.flush();
                totalCnt++;
                if (Math.abs(realLabel - predictLabel) / realLabel < 0.3) {
                    goodCnt++;
                }
                diff += Math.pow(predictLabel - realLabel, 2);
            }
            diff /= data.getSampleNum();
            System.out.println("diff: " + diff);
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Test finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
        System.out.println("good sample number: " + goodCnt);
        System.out.println("total sample number: " + totalCnt);
        System.out.println("good per: " + 1.0 * goodCnt / totalCnt);
        return diff;
    }
}

// End RegressionDemo.java
