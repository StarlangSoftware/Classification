package Classification.Instance;

import Classification.Attribute.Attribute;

import java.util.ArrayList;

public class WordInstance extends Instance{

    private WordInstance previousWord = null;
    private WordInstance nextWord = null;

    public WordInstance(String classLabel, ArrayList<Attribute> attributes) {
        super(classLabel, attributes);
    }

    public WordInstance(String classLabel) {
        super(classLabel);
    }

    public WordInstance(String classLabel, ArrayList<Attribute> attributes, WordInstance previousWord) {
        super(classLabel, attributes);
        this.previousWord = previousWord;
        previousWord.setNextWord(this);
    }

    public WordInstance(String classLabel, WordInstance previousWord) {
        super(classLabel);
        this.previousWord = previousWord;
        previousWord.setNextWord(this);
    }

    public WordInstance getPreviousWord(){
        return previousWord;
    }

    public WordInstance getNextWord(){
        return nextWord;
    }

    public void setPreviousWord(WordInstance previousWord){
        this.previousWord = previousWord;
    }

    public void setNextWord(WordInstance nextWord){
        this.nextWord = nextWord;
    }

    public int numberOfWords(){
        WordInstance current = this;
        int count = 0;
        while (current != null){
            count++;
            current = current.getNextWord();
        }
        return count;
    }

}
