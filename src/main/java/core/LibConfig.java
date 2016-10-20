package core;

import libsvm.svm_parameter;
import libsvm.svm_print_interface;

import java.io.*;
import java.util.Properties;

/**
 *
 * Created by edwardlol on 2016/10/12.
 */
public class LibConfig {
    //~ Inner enum classes -----------------------------------------------------

    public enum Type { CLASSIFICATION, REGRESSION }

    //~ Static fields and initializer ------------------------------------------

    private static LibConfig instance = null;

    private static String propertyFile = "libConfig.properties";

    //~ Instance fields --------------------------------------------------------

    final svm_print_interface svm_print_null = (s) -> {};

    Properties properties = new Properties();

    Type type;

    //~ Constructors -----------------------------------------------------------

    private LibConfig() {
        try (FileInputStream propertiesFile = new FileInputStream(propertyFile)) {
            this.properties.load(propertiesFile);
        } catch (FileNotFoundException fnfe) {
            File file = new File(propertyFile);
            try {
                if (file.createNewFile()) {
                    System.out.println("File is created!");
                    initDefaultConfig();
                } else {
                    System.out.println("File already exists.");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (IOException e) {
            System.out.println("config initialization failed!");
            e.printStackTrace();
        }
    }

    //~ Methods ----------------------------------------------------------------

    /**
     * get the only instance of this class
     * @return the only instance of this class
     */
    static LibConfig getInstance() {
        if (instance == null) {
            instance = new LibConfig();
        }
        return instance;
    }

    /**
     * init the default config properties
     */
    private void initDefaultConfig() {
        setProperty("modelFile", "./results/model");
        setProperty("trainData", "./datasets/train.csv");
        setProperty("testData", "./datasets/test.csv");
    }

    /**
     * get the default param according to the type
     * @return the default param
     */
    svm_parameter getDefaultParam() {
        svm_parameter param = new svm_parameter();
        switch (this.type) {
            case REGRESSION:
                param.svm_type = svm_parameter.EPSILON_SVR;
                break;
            case CLASSIFICATION:
            default:
                param.svm_type = svm_parameter.C_SVC;
        }
        param.kernel_type = svm_parameter.RBF;
        param.C = 1;
        param.eps = 0.001;
        param.p = 0.1;
        param.cache_size = 100;
        return param;
    }

    /**
     * set a property to the property file
     * @param key property key
     * @param value property value
     */
    void setProperty(String key, String value) {
        try (OutputStream output = new FileOutputStream(propertyFile)) {
            this.properties.setProperty(key, value);
            this.properties.store(output, null);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}

// End LibConfig.java
