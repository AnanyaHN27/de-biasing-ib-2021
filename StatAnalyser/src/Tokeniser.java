import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

public class Tokeniser {

    private static ArrayList<String> extract(Path path) throws IOException {
        ArrayList<String> ans = new ArrayList<>();

        String input = Files.readString(path, StandardCharsets.US_ASCII);
        StringBuilder s = new StringBuilder();

        for (int i=0; i<input.length(); i++)
        {
            char curr = input.charAt(i);

            if (curr != '\n')
            {
                s.append(curr);
            }
            else
            {
                if (!s.toString().equals("")) {
                    ans.add(s.toString());
                }
                s = new StringBuilder();
            }
        }
        if (!s.toString().equals("")){
            ans.add(s.toString());
        }

        return ans;
    }

    private static boolean isLetter (char c)
    {
        return ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z');
    }

    public static ArrayList<WordPos> tokenise(Path path) throws IOException {

        ArrayList<WordPos> ans = new ArrayList<>();

        String input = Files.readString(path, StandardCharsets.UTF_8);

        StringBuilder s = new StringBuilder();

        for (int i=0; i<input.length(); i++)
        {
            char curr = input.charAt(i);
            if (isLetter(curr) || curr == '-')
            {
                s.append(Character.toLowerCase(input.charAt(i)));
            }
            else
            {
                if (!s.toString().equals(""))
                {
                    ans.add(new WordPos(i - s.length() + 1, s.toString()));
                }
                s = new StringBuilder();
            }
        }
        if (!s.toString().equals(""))
        {
            ans.add(new WordPos(input.length() - s.length() + 1, s.toString()));
        }

        return ans;
    }

    public static ArrayList<WordPos> filter(ArrayList<WordPos> input, Path wordsP) throws IOException {
        ArrayList<String> words = extract(wordsP);

        ArrayList<WordPos> ans = new ArrayList<>();

        for (WordPos wp: input)
        {
            for(String word: words)
            {
                if (wp.getWord().equals(word))
                {
                    ans.add(wp);
                    break;
                }
            }
        }

        return ans;
    }

    public static ArrayList<WordPos> filterExpressions(Path inputP, Path expressionsP, Path synonymsP) throws IOException {

        ArrayList<String> expressions = extract(expressionsP);
        ArrayList<String> synonymLists = extract(synonymsP);

        String input = Files.readString(inputP, StandardCharsets.UTF_8);

        ArrayList<WordPos> ans = new ArrayList<>();

        for (int i=0; i<input.length(); i++)
        {
            if (!(i == 0 || !isLetter(input.charAt(i - 1)))) {
                continue;
            }
            for(int x=expressions.size() - 1; x >= 0; x--)
            {
                String e = expressions.get(x);

                boolean flag = true;
                for (int j=0; j<e.length(); j++)
                {
                    if (i + j >= input.length() || Character.toLowerCase((input.charAt(i + j))) != e.charAt(j))
                    {
                        flag = false;
                        break;
                    }
                }
                if (flag)
                {
                    if (i + e.length() == input.length() || isLetter(input.charAt(i+e.length()))) {
                        flag = false;
                    }
                }

                if (flag)
                {
                    WordPos wp = new WordPos(i, e);
                    //dot in synonyms file means no synonyms for that expression
                    if (!(synonymLists.get(x).equals("."))) {
                        //synonyms which match the expression on line x of expression dataset
                        String[] synonyms = synonymLists.get(x).split(",");
                        for (String s : synonyms) {
                            wp.addSyn(s);
                        }
                    }
                    i = i + e.length() - 1;
                    ans.add(wp);
                }
            }
        }

        return ans;
    }
}
