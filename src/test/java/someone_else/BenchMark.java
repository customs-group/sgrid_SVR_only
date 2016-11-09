package someone_else;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Stream;

/**
 *
 * Created by edwardlol on 16-10-21.
 */
public class BenchMark {

    /**
     * compare the time cost of reading a file using {@link Stream}
     * and reading a file using {@link BufferedReader}
     */
    @Test
    public void fileReadBench() {
        long streamTime = 0, bufferTime = 0;
        for (int i = 1; i < 10000; i++) {
            streamTime += readStream("./datasets/demo2.train.csv");
        }
        for (int i = 0; i < 10000; i++) {
            bufferTime += readBuffer("./datasets/demo2.train.csv");
        }

        System.out.println("stream time: " + streamTime + "; buffer time: " + bufferTime);
    }

    /**
     * read a file using {@link Stream} and calculate the time cost
     * @param file the file to be read
     * @return the time used to finish reading this file
     */
    private long readStream(String file) {
        long startTime = System.currentTimeMillis();
        try (Stream<String> steam = Files.lines(Paths.get(file))){
            steam.forEach(line -> {
                String[] contents = line.split(",");
                System.out.println(contents.toString());
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        return System.currentTimeMillis() - startTime;
    }

    /**
     * read a file using {@link BufferedReader} and calculate the time cost
     * @param file the file to be read
     * @return the time used to finish reading this file
     */
    private long readBuffer(String file) {
        long startTime = System.currentTimeMillis();
        try (FileReader fr = new FileReader(file);
             BufferedReader br = new BufferedReader(fr)) {
            String line = br.readLine();
            String[] contents = line.split(",");
            while (line != null) {
                contents = line.split(",");
                System.out.println(contents.toString());
                line = br.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return System.currentTimeMillis() - startTime;
    }
}
