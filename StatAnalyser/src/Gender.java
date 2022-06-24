import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

public class Gender {
    public static ArrayList<WordPos> genderMain(ArrayList<WordPos> input) throws IOException {

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

        Path masc = Paths.get(projRoot.toString(),"biased-words", "masculine_words_suffix.txt");
        Path fem = Paths.get(projRoot.toString(),"biased-words", "feminine_words_suffix.txt");

        ArrayList<WordPos> mascCoded = Tokeniser.filter(input, masc);
        for (WordPos wp: mascCoded)
            wp.setGenderM();

        ArrayList<WordPos> femCoded = Tokeniser.filter(input, fem);
        for (WordPos wp: femCoded)
            wp.setGenderF();

        femCoded.addAll(mascCoded);

        return femCoded;
    }
}
