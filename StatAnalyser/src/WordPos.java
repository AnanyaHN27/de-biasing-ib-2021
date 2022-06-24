import java.util.ArrayList;

public class WordPos implements Comparable<WordPos>{
    private Integer position;
    private String word;
    private int length;

    private char gender;
    private char age;
    private char race;
    private char LGBTPl;
    private char biasFlag;
    private String biasName;

    private ArrayList<String> synonyms;

    private Double score;

    WordPos (int pos, String w)
    {
        position = pos;
        word = w;
        length = w.length();

        gender = '-';
        age = '-';
        race = '-';
        biasFlag = '-';
        LGBTPl = '-';

        synonyms = new ArrayList<>();

        score = 1.0;
    }

    void addSyn (String s)
    {
        synonyms.add(s);
    }
    void dropSynonyms()
    {
        synonyms = new ArrayList<>();
    }

    void setScore (double d)
    {
        score = d;
    }

    void setGenderM()
    {
        gender = 'm';
        biasFlag = 'm';
    }
    void setGenderF()
    {
        gender = 'f';
        biasFlag = 'f';
    }
    void setGenderStaticM()
    {
        gender = 'n';
        biasFlag = 'n';
    }
    void setGenderStaticBin()
    {
        gender = 'x';
        biasFlag = 'x';
    }

    void setAgeAntiOlder()
    {
        age = 'o';
        biasFlag = 'o';
    }
    void setAgeAntiYoung()
    {
        age = 'y';
        biasFlag = 'y';
    }
    void setAgeAntiOlderStat()
    {
        age = 'u';
        biasFlag = 'u';
    }
    void setRaceBias()
    {
        race = 't';
        biasFlag = 't';
    }

    void setGeneralBias(char c)
    {
        biasFlag = c;
    }
    void setGeneralBias(String s)
    {
        biasFlag = '0';
        biasName = s;
    }

    void setLGBTPlusBias()
    {
        LGBTPl = 'l';
        biasFlag = 'l';
    }


    Double getScore()
    {
        return score;
    }

    Integer getPosition()
    {
        return position;
    }

    String getWord()
    {
        return word;
    }

    int getLength()
    {
        return length;
    }

    char getGender()
    {
        return gender;
    }

    char getAge()
    {
        return age;
    }

    char getRace()
    {
        return race;
    }

    char getLGBTPl()
    {
        return LGBTPl;
    }

    char getBiasFlag()
    {
        return biasFlag;
    }

    String getBiasName()
    {
        return biasName;
    }

    ArrayList<String> getSynonyms()
    {
        return synonyms;
    }

    @Override
    public int compareTo(WordPos wordPos) {
        return this.getPosition().compareTo(wordPos.getPosition());
    }
}
