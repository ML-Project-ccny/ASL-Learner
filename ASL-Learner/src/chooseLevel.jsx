import React, { useEffect,useState } from "react";
import './chooseLevel.css'
import Dropdown from 'react-bootstrap/Dropdown';
import DropdownButton from 'react-bootstrap/DropdownButton';
import Button from 'react-bootstrap/Button';
import {useNavigate,useLocation} from 'react-router-dom';
import axios from 'axios';

function ChooseLevel(){

    const {state} = useLocation();
    const {email,password} = state
    const [words,setWords] = useState([])
    const [hand,setHand] = useState('right')
    const [complete,setComplete] = useState([])
    //const [newUser, setUser] =useState('')
    

    const navigate = useNavigate();
    
    useEffect( () => {
        getWords()
        console.log(email)
        if(email){
            document.getElementById('acc').setAttribute("hidden", "hidden")
            document.getElementById('acc2').removeAttribute("hidden")
            document.getElementById('acc1').removeAttribute("hidden")
        }
    },[])

    async function getWords(){
        // if user is signed in 
        let arrWords = []
        console.log(email,password)
        if (email !== null){
            const body = {
                email
            }
            try{
                await axios({
                    method: 'post',
                    url: 'http://localhost:5000/words',
                    data: body,
                    headers: {
                        'Content-Type': 'text/plain;charset=utf-8',
                    },
                }).then((res)=>{
                    console.log(res.data)
                    Object.keys(res.data).forEach( (num) =>{
                        console.log(res.data[num])
                        arrWords.push(res.data[num])
                    } )
                    setWords(arrWords)
                    getCompletenes(arrWords)
                })
            }
            catch{
                fetch('http://localhost:5000/allWords')
                .then((response) => response.json())
                .then( (res) => {
                    console.log(res)
                    Object.keys(res).forEach( (num) =>{
                        console.log(res[num])
                        arrWords.push(res[num])
                    } )
                    setWords(arrWords)
                    console.log(arrWords)
                    getCompletenes(arrWords)
                })

                
                document.getElementById('acc').removeAttribute("hidden")
                document.getElementById('acc1').setAttribute("hidden", "hidden")
                document.getElementById('acc2').setAttribute("hidden", "hidden")
                
            }
        }else{
            fetch('http://localhost:5000/allWords')
            .then((response) => response.json())
            .then( (res) => {
                console.log(res)
                Object.keys(res).forEach( (num) =>{
                    console.log(res[num])
                    arrWords.push(res[num])
                } )
                setWords(arrWords)
                console.log(arrWords)
                getCompletenes(arrWords)
            })
        }
    }

    function getCompletenes(arrWords){
        let com = []
        console.log(arrWords)
        arrWords.map( (arr) =>{
            let cnt = 0 
            arr.map( (w) => {
                if(w[1] !== 0){
                    cnt += 1
                }
            })
            com.push([cnt,arr.length])
        })
        console.log(com)
        setComplete(com)
    }

    function changeDisplay(i,j){
        const content = document.getElementById(i)
        const title = document.getElementById(j)
        if (content.style.display === "grid") {
            content.style.display = "none";
            title.style.backgroundColor = 'white'
          } else {
            content.style.display = "grid";
            title.style.backgroundColor = '#d2dce7'
          }
    } 
    function changeHand(x){
        const content = document.getElementById('dropdown-basic-button')
        if (x == 0){
            content.innerHTML = 'Right'
            setHand('right')
        }else{
            content.innerHTML = 'Left'
            setHand('left')
        }
    }

    function navigateGame(word,level){
        let up_word = word.toUpperCase()
        const arr_word = up_word.split("")
        navigate('/game',{state : {word:arr_word,level,username:email,password,hand}});

    }

    function navigateSignIn(){
        navigate('/Login');
    }


    function navigateLogout(){
        // setUser(null)
        // setPwd(null)
        navigate('/Login');
    }

    function navigateInfo(){
        navigate('/Info', {state : {email}});
    }

    return (
        <div className='cont'>
            <Button id='acc' className='account' onClick={navigateSignIn}>Sign In</Button>
            <Button id='acc1' className='account' hidden="hidden" onClick={navigateLogout}>Sign Out</Button>
            <Button id='acc2' className='account' hidden="hidden" onClick={navigateInfo}>User Info</Button>
            <div className='hand'>
                    <h4 className='question'>  Are you left or right handed? </h4>
                    <DropdownButton className='dropDown' id="dropdown-basic-button" title="Right ">
                        <Dropdown.Item onClick={() => changeHand(0)}>Right</Dropdown.Item>
                        <Dropdown.Item onClick={() => changeHand(1)} >Left</Dropdown.Item>
                    </DropdownButton>

                </div>

            {words.map( (arr,i) => {
                return(
                    <div className='level'>
                        <div id={`${i}title`} onClick={() => changeDisplay(i,`${i}title`)} className='title'>
                            <h4>Level {i+1}</h4>
                            <div className="rightTitle">
                                <p className='score'> ({complete[i][0]} / {complete[i][1]})</p>
                                <div className='grayBar'> 
                                    <div className='colorBar' style={{"width":`${complete[i][0] *100 / complete[i][1]}%` }}></div>
                                </div>
                            </div>
                        </div>
                        <div id ={i} class="grid-container">
                        {arr.map( (word) => {
                                return <h4 onClick={() => navigateGame(word[0],i+1)} className={ `word ${ word[1]===0 ? 'wordColor' :'wordColorCompleted'}` }>
                                        <span>{word[0]}</span>
                                    </h4>
                            })}
                        </div>
                    </div>
                )
            })}
        </div>
    )
}

export default ChooseLevel;