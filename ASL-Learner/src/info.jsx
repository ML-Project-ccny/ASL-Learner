import React , {useEffect, useState}from "react";
import './info.css';
import {useLocation, useNavigate} from 'react-router-dom';
import axios from 'axios';

function Info (){

    const {state} = useLocation();
    const {email, password, word, level, hand} = state
    const [wordarr,setWordarr] = useState([])
    const [result,setResult] = useState()
    const [success, setSuccess] = useState(false);

    useEffect( () => {
        if(email){
            setSuccess(true)
            scores()
        }
    },[])

    const navigate = useNavigate();
    

    const navigateChooselevel = () =>{
        navigate('/level',{state:{email, password}});
    }

    const navigateLogin= () =>{
        // setUser("")
        navigate('/Login');
    }

    async function scores(){
        let arrWords = []
        const body = {
            email
        }
        const res = await axios ({
            method: 'POST',
            url: 'http://localhost:5000/words',
            data: body,
            headers: {
                'Content-Type': 'text/plain;charset=utf-8',
            },
        })
        console.log(res.data)
        Object.keys(res.data).forEach( (arr) =>{
            console.log(res.data[arr])
            arrWords.push(res.data[arr])
        } )
        console.log(arrWords)
        setWordarr(arrWords)

        let result_= 0

        arrWords.map((arr, i) => {
            arr.map ((w) => {
                result_ = result_ + w[1] 
            })
            setResult(result_)
        })

    }


    return (
        <>
            {success ? (
                <div className="info">
                    <h1 className="top"> User Info </h1>
                    
                    <div className="contents">
                        
                            {wordarr.map((arr, i) => {
                                if (email){
                                    return(
                                        <div>
                                            <table>
                                                <tr>
                                                    <td> <h4>Level:</h4> </td>
                                                    <td>{i+1}</td>
                                                </tr>
                                                <tr>
                                                    <td> <h4>Score:</h4> </td>
                                                    {/* <td>{result}</td> */}
                                                    <td>{arr.map((w) => {
                                                        return (
                                                            <div>{w[0]} : {w[1]}</div>
                                                        )
                                                    })}</td>
                                                </tr> 
                                            </table>
                                        </div>
                                    )
                                }
                            })}

                            <button class="button-90" >Username: {email}</button>
                            <button onClick={navigateChooselevel} class="button-90" role="button">Level Page</button>
                        
                            <button onClick={navigateLogin} class="button-90" role="button">Log Out</button>

                            
                            
                        
                    </div>
                </div>
            ) : (
                <div className='infos'>
                    <section>
                        <h1>Wrong Request!!</h1>
                        <h5>Try to Log In again</h5>
                        <br />
                        <p>
                            <a href='/'>Go to home Page</a>
                            <br />
                            <a href='/Login'>Go to Login Page</a>
                        </p>
                    </section>
                </div>
                )}
         </>
        
    );
}

export default Info;