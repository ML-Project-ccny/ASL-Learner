import React, { useTransition } from 'react';
import { useRef, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
//import AuthContext from "./context/AuthProvider";

import axios from 'axios';
//const LOGIN_URL = '/auth';

const Login = ({email, setUser, password, setPwd}) => {


    const [emails, setUsers] = useState('');
    const [passwords, setPwds] = useState('');
    const navigate = useNavigate();

    const navigatelevel = () =>{
        navigate('/level',{state:{email,password}});
    }

    async function login() {
        navigatelevel()
        console.log(email, password)

        const body = {
            email,
            password
        }

        console.log(body)
        const res = await axios({
            method: 'GET',
            url: 'http://localhost:5000/user',
            data: body,
            headers: {
                'Content-Type': 'text/plain;charset=utf-8',
            },
        })
        console.log(res.data)
        console.log(res.data['email'])
        if (res.data['email'] === email){
            navigatelevel()
            setSuccess(true)
                
        }
        
    }

    const navigateHome = () =>{
        navigate('/');
    }


    //const { setAuth } = useContext(AuthContext);
    const userRef = useRef();
    const errRef = useRef();

    
    const [errMsg, setErrMsg] = useState('');
    const [success, setSuccess] = useState(false);


    useEffect(() => {
        userRef.current.focus();
    }, [])

    useEffect(() => {
        setErrMsg('');
    }, [email, password])

    const handleSubmit = async (e) => {
        e.preventDefault();
        console.log(email, password);
        setUser('');
        setPwd('');
        setSuccess(true);
        
    }

    return (
        <>
            {success ? (
                <div className='infos'>
                    <section>
                        <h1>You are logged in!</h1>
                        <br />
                        <p>
                            <a href='/game'>Go to Game Page</a>
                            <br />
                            <a href='/info'>Go to Info Page</a>
                        </p>
                    </section>
                </div>
            ) : (
                <div className='infos'>
                    <section>
                        <p ref={errRef} className={errMsg ? "errmsg" : "offscreen"} aria-live="assertive">{errMsg}</p>
                        <h1>Sign In</h1>
                        <form onSubmit={handleSubmit}>
                            <label htmlFor="username">Username:</label>
                            <input
                                type="text"
                                id="username"
                                ref={userRef}
                                autoComplete="off"
                                onChange={(e) => setUser(e.target.value)}
                                value={email}
                                required
                            />

                            <label htmlFor="password">Password:</label>
                            <input
                                type="password"
                                id="password"
                                onChange={(e) => setPwd(e.target.value)}
                                value={password}
                                required
                            />
                            <button onClick={login}>Sign In</button>
                            <button onClick={navigateHome}>Home</button>
                        </form>
                        <p>
                            Need an Account?<br />
                            <span className="line">
                                <a href='/Register'>Sign Up</a>
                            </span>
                        </p>
                    </section>
                </div>
            )}
        </>
    )
}

export default Login