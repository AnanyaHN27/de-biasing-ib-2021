import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.*;
import java.nio.file.Path;
import java.util.ArrayList;

public class ModelAPI {

    public ArrayList<WordPos> parseSyn (Path p) throws IOException, ParseException {

        JSONParser jp = new JSONParser();
        FileReader reader = new FileReader(p.toString());
        Object synObj = jp.parse(reader);
        JSONArray synJArr = (JSONArray) synObj;

        ArrayList<WordPos> ans = new ArrayList<>();

        for (Object o: synJArr)
        {
            JSONObject entry = (JSONObject) o;

            Long posL = (Long) entry.get("start");
            int pos = (int) (long) posL;
            String word = (String) entry.get("word");
            JSONArray jSyn = (JSONArray) entry.get("synonyms");

            WordPos wp = new WordPos(pos, word);

            for (Object strObj: jSyn)
            {
                String synonym = (String) strObj;
                wp.addSyn(synonym);
            }
            String type = (String) entry.get("further_type");
            if (type.equals(""))
            {
                type = (String) entry.get("type");
            }
            //String type = (String) entry.get("type");
            Double sc = (Double) entry.get("score");
            wp.setScore(sc);
            wp.setGeneralBias(type);

            ans.add(wp);
        }

        return ans;
    }



    JSONObject formatJSON (WordPos wp)
    {
        JSONObject wordUnit = new JSONObject();
        wordUnit.put("start", wp.getPosition());
        wordUnit.put("end", wp.getPosition() + wp.getLength() - 1);
        wordUnit.put("word", wp.getWord());
        wordUnit.put("score", wp.getScore());
        if (wp.getBiasFlag() != '0')
        {
            wordUnit.put("type", Character.toString(wp.getBiasFlag()));
        }
        else
        {
            wordUnit.put("type", wp.getBiasName());
        }

        JSONArray synonymsJSON = new JSONArray();
        ArrayList<String> synonyms = wp.getSynonyms();
        for (String syn: synonyms)
            synonymsJSON.add(syn);
        wordUnit.put("synonyms", synonyms);

        return wordUnit;
    }


    JSONArray genPos (ArrayList<Integer> start, ArrayList<Integer> end)
    {
        JSONArray ans = new JSONArray();

        for (int i = 0; i < start.size(); i++)
        {
            JSONArray entry = new JSONArray();
            entry.add(start.get(i));
            entry.add(end.get(i));

            ans.add(entry);
        }

        return ans;
    }
}
