import './App.css';
import React from 'react';
import InputForm from './InputForm';
import ProcessingScreen from './ProcessingScreen';
import WordFeedback from './WordFeedback';


function App() {

  return (
    <div className="App">
         <header className="App-header">
            <p>
                Advert Analyser
            </p>

            <InputForm />

            <br />
      </header>

    </div>
  );

}
export default App;
