import React from 'react';
import Tabs from "./components/Tabs";
import WordFlagOutput from "./components/WordFlagOutput";
import image from './image.png';

class WordFeedback extends React.Component{

      render() {

        if (this.props.show){
           return (
                <div>
                    <Tabs>
                        {this.props.suggestions[0].map((words) =>
                             <div label = {words.word} >
                                <WordFlagOutput flag={words.type} />
                                {words.synonyms.map((suggestion) =>
                                    <p style={{color: '#ffffff'}}>
                                            {suggestion}</p>)}
                             </div>
                             )
                        }
                     </Tabs>
                     <img src={image} height='800'/>
                     <p>
                     This is a graphical representation of the input as seen by the model.
                     </p>
                </div>
           );
        } else {
            return null;
        }
      }
}

export default WordFeedback;
