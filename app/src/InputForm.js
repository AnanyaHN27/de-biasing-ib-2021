import React from 'react';
import ProcessingScreen from './ProcessingScreen';
import WordFeedback from './WordFeedback';
import HighlightedText from './HighlightedText';

{/* This class creates a text box for the job advert entry and a button to submit */}

export default class InputForm extends React.Component{
    constructor(props){
        super(props);
        this.state = {
            value: '',
            genderScore: -1,
            ageScore: -1,
            suggestions: [],
            pressed: false,
            visualisationImage: ''
        };

        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }

     handleChange(event) {
        this.setState({value: event.target.value});
     }


    handleSubmit(event) {   /* Submit button just been pressed */
        this.setState({pressed: true, genderScore: -1});
        event.preventDefault();
        if (this.state.value == '') {
            alert('The text area is empty! Try typing something')
        } else {
            /* Joe's API code*/
            fetch('http://localhost:5000/send_advert', {
                  method: "POST",
                  headers: {
                    "Content-type":"application/json",
                  },
                  body:JSON.stringify(this.state.value)
                }
            ).then(response => {
                  return response.json()
                })
            .then(json => {

              this.setState({genderScore: json['genderScore'], femScore: json['femScore'], mascScore: json['mascScore'], ageScore: json['ageScore'], oldScore: json['oldScore'], youngScore: json['youngScore'], suggestions: json['suggestions']});

            })

        }

      }

    render() {
        return (
            <div>
                <form onSubmit={this.handleSubmit}>
                    {/* input textbox */}
                    <textarea name="adtext" rows="20" cols="100" placeholder="Type or copy your job advert here" onChange={this.handleChange}/>

                    {/* submit button */}
                    <input type="submit" value="Submit" size="100" />
                </form>

                <ProcessingScreen show={this.state.pressed} genderScore={this.state.genderScore} femScore={this.state.femScore} mascScore={this.state.mascScore} ageScore={this.state.ageScore} oldScore={this.state.oldScore} youngScore={this.state.youngScore} />
                <HighlightedText show={this.state.genderScore >= 0} suggestions={this.state.suggestions} text={this.state.value} />
                <WordFeedback show={this.state.genderScore >= 0} suggestions={this.state.suggestions} />
            </div>
        );
    }
}
