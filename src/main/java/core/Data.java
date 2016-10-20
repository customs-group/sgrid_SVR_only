package core;

import libsvm.*;
import util.JDBCUtil;

import java.io.*;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.List;
import java.util.Vector;

/**
 *
 * Created by edwardlol on 16/7/7.
 */
public class Data {
    //~ Instance fields --------------------------------------------------------

    private int sampleNum = 0;
    private int featureNum = 0;
    private double scaleUpperBound = 1.0d;
    private double scaleLowerBound = -1.0d;

    private Vector<Double> labels = new Vector<>();
    private Vector<svm_node[]> originalSamples = new Vector<>();
    private Vector<svm_node[]> scaledSamples = null;

    //~ Constructors -----------------------------------------------------------

    public Data() {}

    /**
     * for cross validation
     * @param samples samples
     * @param labels labels
     */
    public Data(List<svm_node[]> samples, List<Double> labels) {
        assert samples.size() == labels.size();
        this.sampleNum = samples.size();
        this.originalSamples = new Vector<>(samples);
        this.labels = new Vector<>(labels);
        this.featureNum = this.originalSamples.get(0).length;
    }

    //~ Methods ----------------------------------------------------------------

    /**
     * init the dataset from a csv file
     * @param file csv file name
     * @return this
     */
	public Data readDataFromCSVFile(String file) {
        long startTime = System.currentTimeMillis();
        try (FileReader fr = new FileReader(file);
             BufferedReader br = new BufferedReader(fr)) {
            String line = br.readLine();
            String[] contents = line.split(",");
            this.featureNum = contents.length - 1;

            while (line != null) {
                contents = line.split(",");
                // check data format
                int lenth = contents.length - 1;
                if (lenth != this.featureNum) {
                    System.err.println("data format not aligned");
                    throw new Exception("data format error");
                }
                // y, x1, x2,...,xn
                svm_node[] sample = new svm_node[this.featureNum];
                for (int i = 0; i < this.featureNum; i++) {
                    sample[i] = new svm_node();
                    sample[i].index = i + 1;
                    sample[i].value = Double.valueOf(contents[i + 1]);
                }
                this.originalSamples.add(sample);
                this.labels.add(Double.valueOf(contents[0]));
                line = br.readLine();
            }
            this.sampleNum = this.originalSamples.size();
            this.scaledSamples = this.originalSamples;
            // end data preparation
            System.out.println("Data preparation done in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
            System.out.println("Read " + this.getSampleNum() + " samples in total");
        } catch (Exception e) {
            System.out.println("Data preparation failed!");
            e.printStackTrace();
        }
        return this;
	}

    /**
     * init the dataset from a DBMS
     * @param url the url of the DBMS
     * @param username username
     * @param password password
     */
    @SuppressWarnings("unused")
	public Data readDataFromDB(String url, String tableName, String username, String password) {
        JDBCUtil jdbcUtil = JDBCUtil.getInstance();
        jdbcUtil.dbms = JDBCUtil.DBMS.ORACLE;
        try (Connection con = jdbcUtil.getConnection(url, username, password);
             Statement stmt = con.createStatement();
             ResultSet rs = stmt.executeQuery("SELECT * FROM " + tableName + ";")) {

            int columnCount = rs.getMetaData().getColumnCount();
            this.featureNum = columnCount - 1;

            while (rs.next()) {
                this.labels.add(rs.getDouble(0));
                svm_node[] sample = new svm_node[this.featureNum];
                for (int i = 0; i < this.featureNum; i++) {
                    sample[i] = new svm_node();
                    sample[i].index = i + 1;
                    sample[i].value = rs.getDouble(i + 1);
                }
                this.originalSamples.add(sample);
            }
        } catch (SQLException se) {
            System.out.println("DBMS connection failed!");
            se.printStackTrace();
        }
        return this;
    }

    /**
     * record data to file
     * for debug usage, when you want to use standalone libsvm to validate the result
     * @param fileName file name to store data
     * @param type type of data to be recorded, original or scaled
     */
    public void recordData(String fileName, String type) {
        long startTime = System.currentTimeMillis();
        String _fileName;
        Vector<svm_node[]> _set;
		/* set file name for record */
        switch (type.toLowerCase()) {
            case "original":
                _fileName = fileName + ".original.txt";
                _set = this.originalSamples;
                break;
            case "scaled":
                _fileName = fileName + ".scaled.txt";
                _set = this.scaledSamples;
                break;
            default:
                System.out.println("wrong data type, recording original set");
                _fileName = fileName + ".original.txt";
                _set = this.originalSamples;
        }
        try (FileWriter fw = new FileWriter(_fileName);
             BufferedWriter bw = new BufferedWriter(fw)) {

            for (int i = 0; i < this.sampleNum; i++) {
                bw.write(this.labels.get(i) + " ");
                svm_node[] sample = _set.get(i);
                for (int j = 0; j < this.featureNum; j++) {
                    bw.write(sample[j].index + ":" + sample[j].value + " ");
                }
                bw.write("\n");
                bw.flush();
            }
            System.out.println("Data record done in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
            System.out.println("see " + _fileName);
        } catch (IOException e) {
            System.out.println("Data record failed!");
            e.printStackTrace();
        }
    }

    /**
     * automatically scale the data according to the min/max value of each column
     * @return a scale_param is a double[][] that contains the min/max value of each column
     */
    public double[][] scaleTrainingData() {
        this.scaledSamples = new Vector<>();
		/* step 0: initiate scale param */
        double[][] scale_param = new double[this.featureNum + 1][2];
        scale_param[0][0] = this.scaleUpperBound;
        scale_param[0][1] = this.scaleLowerBound;
		/* step 1: initiate feature bound */
        double[] feature_max = new double[this.featureNum];
        double[] feature_min = new double[this.featureNum];
        for(int i = 0; i < this.featureNum; i++) {
            feature_max[i] = -Double.MAX_VALUE;
            feature_min[i] = Double.MAX_VALUE;
        }
		/* step 2: find out min/max value */
        for (int i = 0; i < this.sampleNum; i++) {
            for (int j = 0; j < this.featureNum; j++) {
                feature_max[j] = Math.max(feature_max[j], this.originalSamples.get(i)[j].value);
                feature_min[j] = Math.min(feature_min[j], this.originalSamples.get(i)[j].value);
                scale_param[j + 1][0] = feature_max[j];
                scale_param[j + 1][1] = feature_min[j];
            }
        }
		/* step 3: scale */
        for (int i = 0; i < this.sampleNum; i++) {
            svm_node[] originalSample = this.originalSamples.get(i);
            svm_node[] scaledSample = new svm_node[this.featureNum];
            for (int j = 0; j < this.featureNum; j++) {
                scaledSample[j] = new svm_node();
                scaledSample[j].index = originalSample[j].index;
                if (originalSample[j].value == feature_min[j]) {
                    scaledSample[j].value = this.scaleLowerBound;
                } else if (originalSample[j].value == feature_max[j]) {
                    scaledSample[j].value = this.scaleUpperBound;
                } else {
                    scaledSample[j].value = this.scaleLowerBound
                            + ((originalSample[j].value - feature_min[j])
                            / (feature_max[j] - feature_min[j])
                            * (this.scaleUpperBound - this.scaleLowerBound));
                }
            }
            this.scaledSamples.add(scaledSample);
        }
        return scale_param;
    }

    /**
     * scale test data
     * @param scaleParam returned by {@link #scaleTrainingData()}
     */
    public void scaleTestData(double[][] scaleParam) {
        this.scaledSamples = new Vector<>();
		/* step 1: initiate feature bound */
        this.scaleUpperBound = scaleParam[0][0];
        this.scaleLowerBound = scaleParam[0][1];
        /* step 2: read scale param */
        double[] feature_max = new double[this.featureNum];
        double[] feature_min = new double[this.featureNum];
        for(int i = 0; i < this.featureNum; i++) {
            feature_max[i] = scaleParam[i + 1][0];
            feature_min[i] = scaleParam[i + 1][1];
        }
		/* step 3: scale */
        for (int i = 0; i < this.sampleNum; i++) {
            svm_node[] sample = this.originalSamples.get(i);
            svm_node[] scaled_sample = new svm_node[this.featureNum];
            for (int j = 0; j < this.featureNum; j++) {
                scaled_sample[j] = new svm_node();
                scaled_sample[j].index = sample[j].index;
                if (sample[j].value == feature_min[j]) {
                    scaled_sample[j].value = this.scaleLowerBound;
                } else if (sample[j].value == feature_max[j]) {
                    scaled_sample[j].value = this.scaleUpperBound;
                } else {
                    scaled_sample[j].value = this.scaleLowerBound
                            + ((sample[j].value - feature_min[j])
                            / (feature_max[j] - feature_min[j])
                            * (this.scaleUpperBound - this.scaleLowerBound));
                }
            }
            this.scaledSamples.add(scaled_sample);
        }
    }

    /**
     * make the labels 1 or -1
     * for classification
     */
    @SuppressWarnings("unused")
    public void normalizeLabel() {
        this.labels.forEach(label -> label = label <= 0 ? -1.0d : 1.0d);
    }

    //~ Getter/setter methods --------------------------------------------------

    public Vector<svm_node[]> getDataSet(String type) {
        switch (type.toLowerCase()) {
            case "original":
                return this.originalSamples;
            case "scaled":
                if (this.scaledSamples != null) {
                    return this.scaledSamples;
                } else {
                    if (SVMLib.DEBUG) {
                        System.out.println("dataset not scaled yet, original data set returned");
                    }
                    return this.originalSamples;
                }
            default:
                if (SVMLib.DEBUG) {
                    System.out.println("wrong data type, original data set returned");
                }
                return this.originalSamples;
        }
    }
    public Vector<Double> getLabels() {
        return this.labels;
    }
    public int getSampleNum() {
        return this.sampleNum;
    }
    public int getFeatureNum() {
        return this.featureNum;
    }
    public void setScaleBound(double scaleUpperBound, double scaleLowerBound) {
        this.scaleUpperBound = scaleUpperBound;
        this.scaleLowerBound = scaleLowerBound;
    }
}

// End Data.java
