import React from 'react';

export default class WordFeedback extends React.Component{

    render(){
        const flag = this.props.flag;
        const style = {
            'font-size': 15
        };
        let text = '';

        switch(flag) {
            case 'a': text = "This word was highlighted as it has age bias relating to health."; break;
            case 'b': text = "This word was highlighted as it has age bias relating to technical ability"; break;
            case 'c': text = "This word was highlighted asit has age bias relating to personality"; break;
            case 'd': text = "This word was highlighted as it has gender bias, specifically feminine bias relating to motherhood."; break;
            case 'e': text = "This word was highlighted because it has gender bias, specifically feminine bias relating to cooperation."; break;
            case 'g': text = "This word was highlighted as it has gender bias, specifically feminine bias relating to being gentle."; break;
            case 'h': text = "This word was highlighted as it has gender bias, specifically masculine bias relating to being dominant."; break;
            case 'i': text = "This word was highlighted as it has gender bias, specifically masculine bias relating to strength."; break;
            case 'm': text = "This word was highlighted as it has gender bias specifically masculine bias."; break;
            case 'f': text = "This word was highlighted as it has gender bias specifically feminine bias."; break;
            case 'o': text = "This word was highlighted as it has age bias specifically against young people."; break;
            case 'y': text = "This word was highlighted as it has age bias specifically against old people."; break;
            case 'n': text = "This word was highlighted as it has gender bias specifically masculine bias (as picked up by the statistical analysis)."; break;
            case 'u': text = "This word was highlighted as it has age bias specifically against older people."; break;
            case 't': text = "This word was highlighted as it has racial, ethnicity or immigration bias."; break;
            case 'l': text = "This word was highlighted as it has sexual orientation bias."; break;
            default : text = "";
        }
        return (
            <p style={style}>
                {text} Suggested synonyms:
            </p>
        )
    }
}
