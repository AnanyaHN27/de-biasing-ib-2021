import React from 'react';

class HighlightedText extends React.Component{

    render() {
        const text = this.props.text;
        if (this.props.show){
            let pointers = this.props.suggestions[1].map((words) => words.end);
            pointers = [0].concat(pointers);
            pointers = pointers.concat([text.length])
            /* pointers is now an array of the end points of every word and starts with 0 */
            let i=0;

            return (
                <>
                <p style={{'font-size': '15px'}}>
                    Highlighted below are the 10 words that were detected by the model to have the highest amount of bias. By clicking on the word below the text you will be presented with synonyms.
                </p>
                <p class="description" style={{'padding-left': 50, 'padding-right': 50, 'white-space': 'pre-line'}}>
                    {this.props.suggestions[1].map((words) =>
                            <>
                                {text.substring(pointers[i++], words.start)}
                                <em>{text.substring(words.start,words.end)}</em>
                            </>
                     )}

                     {text.substring(pointers[i++], text.length)}
                </p>
                </>
            );
        } else {
            return null;
        }
    }
}

export default HighlightedText;
