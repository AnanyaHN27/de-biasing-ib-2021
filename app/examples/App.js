import React from 'react';
import logo from './logo.svg';
import './App.css';

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {jobText: '',
                  genderScore: -1,
                  ageScore: -1,
                  suggestions: [],
    };
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleSubmit(event) {
    event.preventDefault(); /* this is really important! stops the page refreshing and ruining everything */
    fetch('http://localhost:5000/send_advert', {
      method: "POST",
      headers: {
        "Content-type":"application/json",
      },
      body:JSON.stringify(this.state.jobText)
    }
    ).then(response => {
      return response.json()
    })
    .then(json => {
      /* i think you can do this in one call of setState? not sure though */
      this.setState({genderScore: json['genderScore']});
      this.setState({ageScore: json['ageScore']});
      this.setState({suggestions: json['suggestions']});
    })
    }

  handleChange(event) {
    this.setState({jobText: event.target.value});
  }

  render() {
    return (
      <div className="App">
      
        <form onSubmit={this.handleSubmit}>
          <input type="text" name="jobAdvert" onChange={this.handleChange}/>
          <input type="submit" value="Submit"/>
        </form>

      {this.state.genderScore >= 0 &&
        <>
          <h1>Gender Score: {this.state.genderScore} </h1>
          <h1>Age Score: {this.state.ageScore} </h1>
          {/* suggestions will be p unreadable in this output, but you get the idea */}
          <h1>Suggestions: {JSON.stringify(this.state.suggestions)} </h1> 
        </>
      }
      </div>
    );
  }
}

export default App;
