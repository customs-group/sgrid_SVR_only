package someone_else;

import com.google.common.collect.Maps;
import core.LibConfig;
import core.SVMLib;
import libsvm.svm_model;
import libsvm.svm_node;
import org.junit.Test;

import java.io.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;

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
        SVMLib svmLib = SVMLib.getInstance().setType(LibConfig.Type.REGRESSION).initDataFromFile("./datasets/train.csv");

        // uncomment this line to do cross validation and utilize the svm_param
        // caution: cost a lot of time
//        svmLib.svm_param = svmLib.updateParam();

        svm_model model = svmLib.train();
        regressionResult(model, "./datasets/test.csv", "./results/result.txt");
    }


    /**
     * regression of weather data and defect count
     */
    @Test
    public void regression2() {
        SVMLib svmLib = SVMLib.getInstance().setType(LibConfig.Type.REGRESSION).initDataFromFile("./datasets/demo2.train.csv");

        svm_model model = svmLib.train();
        regressionResult(model, "./datasets/demo2.test.csv", "./results/result2.txt");
    }

    //~ Helper methods ---------------------------------------------------------

    /**
     * convert a string array to a double array
     * @param from source string array
     * @return destination double array
     */
    private double[] convert(String[] from) {
        double[] to = new double[from.length];
        for (int i = 0; i < from.length; i++) {
            to[i] = Double.valueOf(from[i]);
        }
        return to;
    }

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
     * conbine the two parts of weather data and defect data
     * and output to train data and test data
     */
    @Test
    public void combine() {
        Map<LocalDate, String> defcnt = readPart1("./datasets/defcnt.csv");
        Map<LocalDate, Integer[]> weather = readPart2("./datasets/weather.csv");

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
        assert samples.size() == labels.size();

        double trainingRatio = 0.9d;
        int trainingIndex = (int)Math.round(samples.size() * trainingRatio);

        Vector<svm_node[]> trainSamples = new Vector<>(samples.subList(0, trainingIndex));
        Vector<Double> trainLabels = new Vector<>(labels.subList(0, trainingIndex));
        try (FileWriter fw = new FileWriter("./datasets/demo2.train.csv");
             BufferedWriter bw = new BufferedWriter(fw)){
            for (int i = 0; i < trainSamples.size(); i++) {
                bw.append(trainLabels.get(i).toString());
                bw.append(",");
                svm_node[] sample = trainSamples.get(i);
                for (int j = 0; j < sample.length; j++) {
                    bw.append(String.valueOf(sample[j].value));
                    if (j < sample.length - 1) {
                        bw.append(",");
                    }
                }
                bw.append("\n");
                bw.flush();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        Vector<svm_node[]> testSamples = new Vector<>(samples.subList(trainingIndex, samples.size()));
        Vector<Double> testLabels = new Vector<>(labels.subList(trainingIndex, samples.size()));
        try (FileWriter fw = new FileWriter("./datasets/demo2.test.csv");
             BufferedWriter bw = new BufferedWriter(fw)) {
            for (int i = 0; i < testSamples.size(); i++) {
                bw.append(testLabels.get(i).toString());
                bw.append(",");
                svm_node[] sample = testSamples.get(i);
                for (int j = 0; j < sample.length; j++) {
                    bw.append(String.valueOf(sample[j].value));
                    if (j < sample.length - 1) {
                        bw.append(",");
                    }
                }
                bw.append("\n");
                bw.flush();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    private double regressionResult(svm_model model, String testFile, String resultFile) {
        long startTime = System.currentTimeMillis();
        double diff = 0.0d;
        int totalCnt = 0, goodCnt = 0;

        try (FileReader fr = new FileReader(testFile);
             BufferedReader br = new BufferedReader(fr);
             FileWriter fw = new FileWriter(resultFile);
             BufferedWriter bw = new BufferedWriter(fw)){

            String line = br.readLine();
            while(line != null) {
                totalCnt++;
                String[] content = line.split(",");
                double realLabel = Double.valueOf(content[0]);
                double[] sample = convert(Arrays.copyOfRange(content, 1, content.length));

                double predictLabel = SVMLib.predict(sample, model);
                bw.write("predict label: " + predictLabel + "; real label: " + realLabel + "; ");
                for (int i = 0; i < sample.length; i ++) {
                    bw.write((i + 1) + ":" + sample[i] + " ");
                }
                bw.write("\n");
                bw.flush();
                double _diff = Math.abs(predictLabel - realLabel) / realLabel;
                if (_diff < 0.3) {
                    goodCnt++;
                }
                diff += _diff;
                line = br.readLine();
            }
            diff /= totalCnt;
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
