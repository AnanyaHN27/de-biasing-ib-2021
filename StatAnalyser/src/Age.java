import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

public class Age {
    public static ArrayList<WordPos> ageMain(ArrayList<WordPos> input) throws IOException {

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

        Path older = Paths.get(projRoot.toString(),"biased-words", "ageism_anti-older.txt");
        Path young = Paths.get(projRoot.toString(),"biased-words", "ageism_anti-young.txt");

        ArrayList<WordPos> antiOlderCoded = Tokeniser.filter(input, older);
        for (WordPos wp: antiOlderCoded)
            wp.setAgeAntiOlder();

        ArrayList<WordPos> antiYoungCoded = Tokeniser.filter(input, young);
        for (WordPos wp: antiYoungCoded)
            wp.setAgeAntiYoung();

        antiOlderCoded.addAll(antiYoungCoded);

        return antiOlderCoded;
    }
}
