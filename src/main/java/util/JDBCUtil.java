package util;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.Properties;

/**
 *
 * Created by edwardlol on 2016/10/12.
 */
public class JDBCUtil {
    //~ Enum class -------------------------------------------------------------

    public enum DBMS {
        ORACLE, MYSQL, DERBY, SQLSERVER, DB2, INFORMIX, SYBASE, POSTGRESQL
    }

    //~ Static fields/initializers ---------------------------------------------

    private static JDBCUtil instance = null;

    //~ Instance fields --------------------------------------------------------

    public DBMS dbms;

    public String dbName;

    //~ Constructors -----------------------------------------------------------

    private JDBCUtil() {}

    //~ Methods ----------------------------------------------------------------

    /**
     * get the only instance of this class
     * @return the only instance of this class
     */
    public static JDBCUtil getInstance() {
        if (instance == null) {
            instance = new JDBCUtil();
        }
        return instance;
    }

    /**
     * regist the jdbc driver
     */
    public void registJDBC() {
        try {
            switch (this.dbms) {
                case ORACLE:
                    Class.forName("oracle.jdbc.driver.OracleDriver");
                    break;
                case MYSQL:
                    Class.forName("com.mysql.jdbc.Driver");
                    break;
                case DERBY:
                    Class.forName("org.apache.derby.jdbc.EmbeddedDriver");
                    break;
                case SQLSERVER:
                    Class.forName("com.microsoft.jdbc.sqlserver.SQLServerDriver");
                    break;
                case DB2:
                    Class.forName("com.ibm.db2.jdbc.app.DB2Driver");
                    break;
                case INFORMIX:
                    Class.forName("com.informix.jdbc.IfxDriver");
                    break;
                case SYBASE:
                    Class.forName("com.sybase.jdbc.SybDriver");
                    break;
                case POSTGRESQL:
                    Class.forName("org.postgresql.Driver");
                    break;
            }
        } catch (ClassNotFoundException ex) {
            System.out.println("Error: unable to load driver class!");
            System.exit(1);
        }
    }

    /**
     * get the connection of the DBMS
     * the password should be encrypted before this
     * @param url the url to the DBMS
     * @param username username
     * @param password password
     * @return the connection of the DBMS
     * @throws SQLException
     */
    public Connection getConnection(String url, String username, String password) throws SQLException {
        Connection conn = null;

        registJDBC();

        Properties connectionProps = new Properties();
        connectionProps.put("user", username);
        connectionProps.put("password", password);

        switch (this.dbms) {
            case ORACLE:
                conn = DriverManager.getConnection(
                        "jdbc:" + this.dbms.toString().toLowerCase() + "thin:@" + url + "/",
                        connectionProps);
                break;
            case MYSQL:
                conn = DriverManager.getConnection(
                        "jdbc:" + this.dbms.toString().toLowerCase() + "://" + url + "/",
                        connectionProps);
                break;
            case DERBY:
                conn = DriverManager.getConnection(
                        "jdbc:" + this.dbms.toString().toLowerCase() + ":" + url + ";create=true",
                        connectionProps);
                break;
            case SQLSERVER:
                conn = DriverManager.getConnection(
                        "jdbc:" + this.dbms.toString().toLowerCase() + "://" + url + "/",
                        connectionProps);
                break;
            case DB2:
                conn = DriverManager.getConnection(
                        "jdbc:" + this.dbms.toString().toLowerCase() + ":" + url + "/",
                        connectionProps);
                break;
            case INFORMIX:
                conn = DriverManager.getConnection(
                        "jdbc:" + this.dbms.toString().toLowerCase() + "://" + url + "/",
                        connectionProps);
                break;
            case SYBASE:
                conn = DriverManager.getConnection(
                        "jdbc:" + this.dbms.toString().toLowerCase() + ":Tds:" + url + "/",
                        connectionProps);
                break;
            case POSTGRESQL:
                conn = DriverManager.getConnection(
                        "jdbc:" + this.dbms.toString().toLowerCase() + "://" + url + "/",
                        connectionProps);
                break;
        }

        System.out.println("Connected to database");
        return conn;
    }
}

// End JDBCUtil.java
