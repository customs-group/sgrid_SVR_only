package core;

import libsvm.svm_parameter;
import libsvm.svm_print_interface;

/**
 *
 * Created by edwardlol on 2016/10/12.
 */
public class LibConfig {
    //~ Static fields and initializer ------------------------------------------

    // TODO: 2016/10/10 solve this shit
    public static final svm_print_interface svm_print_null = new svm_print_interface() {
        public void print(String s) {}
    };

    public static svm_parameter getDefaultParam(String svmType) {
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

    //~ Constructors -----------------------------------------------------------

    // Suppress default constructor for noninstantiability
    private LibConfig() {}

}

// End LibConfig.java
