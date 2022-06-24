import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

public class LGBTQPlus {
    public static ArrayList<WordPos> LGBTPlusMain(Path input) throws IOException {
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

        Path expressions = Paths.get(projRoot.toString(),"biased-words", "lgbtqia+_bias.txt");
        Path synonymLists = Paths.get(projRoot.toString(),"biased-words", "lgbtqia+_bias_synonyms.txt");

        ArrayList<WordPos> filtered = Tokeniser.filterExpressions(input, expressions, synonymLists);

        for (WordPos wp: filtered)
            wp.setLGBTPlusBias();

        return filtered;
    }
}