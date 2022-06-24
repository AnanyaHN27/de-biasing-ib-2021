import React from 'react';


class ProcessingScreen extends React.Component{
    constructor(props){
        super(props);
    }

      render() {
            if (!this.props.show){
                return null;

            } else if (this.props.genderScore < 0) {
                return <h1>Processing...</h1>;

            } else{

                return (
                    <>
                        <h1 style={{color:"#ffffff",'font-size': 40, 'float':'left', width: '50%'}}>
                            Gender Score: {this.props.genderScore}
                        </h1>
                        <h1 style={{color:"#ffffff",'font-size': 40, 'float':'right', width: '50%'}}>
                            Age Score: {this.props.ageScore}
                        </h1>
                        <h3 style={{color:"#aaaaaa",'font-size': 15, 'float':'left', width: '50%', 'margin-top': '-20px'}}>
                            Masculine bias: {this.props.mascScore} Feminine bias: {this.props.femScore}
                        </h3>
                        <h3 style={{color:"#aaaaaa",'font-size': 15, 'float':'right', width: '50%', 'margin-top': '-20px'}}>

                            Old bias: {this.props.oldScore} Young bias: {this.props.youngScore}
                        </h3>

                        <p style={{'font-size': '15px'}}>
                            The above scores out of 100 represent how biased your advert is, with a higher score meaning your advert is more neutral.
                        </p>
                    </>
                );
            }
      }
}

export default ProcessingScreen;
