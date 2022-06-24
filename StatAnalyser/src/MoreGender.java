import jdk.swing.interop.SwingInterOpUtils;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

public class MoreGender {
    public static ArrayList<WordPos> genderBinary(Path input) throws IOException {
        Path projRoot = Paths.get("").toAbsolutePath();

        while (!projRoot.getFileName().toString().equals("de-biasing-ib-2021")) {
            try {
                projRoot = projRoot.getParent();
            }
            // in case we run forever and get to very top directory
            catch(NullPointerException e) {
                System.err.println("Uh oh, you're not running this from a folder within the project!");
            }
        }

        Path expressions = Paths.get(projRoot.toString(),"biased-words", "gender_binary_expressions.txt");
        Path synonymLists = Paths.get(projRoot.toString(),"biased-words", "gender_binary_synonyms.txt");

        ArrayList<WordPos> filtered = Tokeniser.filterExpressions(input, expressions, synonymLists);

        for (WordPos wp: filtered)
            wp.setGenderStaticBin();

        return filtered;
    }

    public static ArrayList<WordPos> masculine(Path input) throws IOException {
        Path projRoot = Paths.get("").toAbsolutePath();

        while (!projRoot.getFileName().toString().equals("de-biasing-ib-2021")) {
            try {
                projRoot = projRoot.getParent();
            }
            // in case we run forever and get to very top directory
            catch(NullPointerException e) {
                System.err.println("Uh oh, you're not running this from a folder within the project!");
            }
        }

        Path expressions = Paths.get(projRoot.toString(),"biased-words", "masculine_expressions.txt");
        Path synonymLists = Paths.get(projRoot.toString(),"biased-words", "masculine_expression_synonyms.txt");

        ArrayList<WordPos> filtered = Tokeniser.filterExpressions(input, expressions, synonymLists);

        for (WordPos wp: filtered)
            wp.setGenderStaticM();

        return filtered;
    }
}
