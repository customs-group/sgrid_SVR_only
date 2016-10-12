package core;

import libsvm.svm_parameter;
import libsvm.svm_print_interface;

import java.io.*;
import java.util.Properties;

/**
 *
 * Created by edwardlol on 2016/10/12.
 */
class LibConfig {
    //~ Static fields and initializer ------------------------------------------

    // TODO: 2016/10/10 solve this shit
    static final svm_print_interface svm_print_null = new svm_print_interface() {
        public void print(String s) {}
    };

    private static String propertyFile = "libConfig.properties";

    static Properties properties = new Properties();
    static {
        try (FileInputStream propertiesFile = new FileInputStream(propertyFile)) {
            properties.load(propertiesFile);
        } catch (FileNotFoundException fnfe) {
            File file = new File(propertyFile);
            try {
                if (file.createNewFile()) {
                    System.out.println("File is created!");
                } else {
                    System.out.println("File already exists.");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //~ Constructors -----------------------------------------------------------

    // Suppress default constructor for noninstantiability
    private LibConfig() {}

    //~ Static methods ---------------------------------------------------------

    static svm_parameter getDefaultParam(String svmType) {
        svm_parameter param = new svm_parameter();
        switch (svmType.toLowerCase()) {
            case "svr":
                param.svm_type = svm_parameter.EPSILON_SVR;
                break;
            case "svm":
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

    static void addProperty(String key, String value) {
        try (OutputStream output = new FileOutputStream(propertyFile)) {
            properties.setProperty(key, value);
            properties.store(output, null);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

// End LibConfig.java
