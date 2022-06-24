import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.ParseException;

public class Main {

    static ArrayList<WordPos> merge (ArrayList<WordPos> statBiased, ArrayList<WordPos> modelBiased)
    {
        ArrayList<WordPos> ans = new ArrayList<>();

        int posM = 0;

        for (WordPos wordPos : statBiased)
        {
            while (posM < modelBiased.size() && modelBiased.get(posM).getPosition() + modelBiased.get(posM).getLength() - 1 < wordPos.getPosition())
            {
                ans.add(modelBiased.get(posM));
                posM++;
            }
            ans.add(wordPos);

            while (posM < modelBiased.size() && wordPos.getPosition() >= modelBiased.get(posM).getPosition() + modelBiased.get(posM).getLength() - 1)
            {
                posM++;
            }
        }

        while (posM < modelBiased.size())
        {
            ans.add(modelBiased.get(posM));
            posM++;
        }

        return ans;
    }

    public static void main(String[] args) throws IOException, ParseException {

        int N = Integer.parseInt(args[2]);
        Path p = Paths.get(System.getProperty("user.dir"), args[0]);

        ArrayList<WordPos> input = Tokeniser.tokenise(p);

        //ArrayList<WordPos> genderBiased = Gender.genderMain(input);
        //ArrayList<WordPos> ageBiased = Age.ageMain(input);
        ArrayList<WordPos> statGenderBinBiased = MoreGender.genderBinary(p);
        ArrayList<WordPos> statGenderMascBiased = MoreGender.masculine(p);
        ArrayList<WordPos> statAgeBiased = MoreAge.ageism(p);
        ArrayList<WordPos> raceBiased = Race.raceMain(p);
        ArrayList<WordPos> LGBTPlusBiased = LGBTQPlus.LGBTPlusMain(p);

        ArrayList<WordPos> statBiased = statGenderBinBiased;
        statBiased.addAll(statGenderMascBiased);
        statBiased.addAll(statAgeBiased);
        statBiased.addAll(raceBiased);
        statBiased.addAll(LGBTPlusBiased);


        Collections.sort(statBiased);



        Path synP = Paths.get(System.getProperty("user.dir"), args[1]);
        ModelAPI API = new ModelAPI();
        ArrayList<WordPos> modelBiased = API.parseSyn(synP);

        Collections.sort(modelBiased);


        ArrayList<WordPos> biased = merge(statBiased, modelBiased);



        biased.sort(Comparator.comparing(WordPos::getScore).thenComparing(WordPos::getWord).reversed());
        ArrayList<WordPos> finalArr = new ArrayList<>();
        ArrayList<Integer> startPos = new ArrayList<>();
        ArrayList<Integer> endPos = new ArrayList<>();

        int finalSize = 0;
        for (WordPos pos : biased) {

            startPos.add(pos.getPosition());
            endPos.add(pos.getPosition() + pos.getLength() - 1);

            if (finalSize == 0 || !finalArr.get(finalSize - 1).getWord().equals(pos.getWord())) {
                finalArr.add(pos);
                finalSize++;
                if (finalSize == N) {
                    break;
                }
            }
        }
        Collections.sort(finalArr);


        JSONArray output = new JSONArray();

        for (WordPos wordPos : finalArr) {
            JSONObject jo = API.formatJSON(wordPos);
            output.add(jo);
        }

        Files.write(Paths.get("output"), output.toJSONString().getBytes());


        JSONArray positions = API.genPos(startPos, endPos);
        Files.write(Paths.get("positions"), positions.toJSONString().getBytes());
    }
}
