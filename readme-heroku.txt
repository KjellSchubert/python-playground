Goal was running this an asyncio Python server with a free Heroku account experimentally.

My steps were:

https://devcenter.heroku.com/articles/getting-started-with-python
install python 3.4.1
>pip install virtualenv
>virtualenv venv

So this virtualenv creates a folder venv, roughly corresponding to a nodejs local npm download dir (see http://docs.python-guide.org/en/latest/starting/install/win):

>venv\scripts\activate.bat
>venv\scripts\deactivate.bat

Cool! The Heroku demo app uses Django, but according to https://devcenter.heroku.com/categories/python other python server frameworks can be used (I want aiohttp atm).

Installed Heroku Toolbelt. Then:
>heroku login
>pip install aiohttp
>pip freeze > requirements.txt
optional >pip install -r requirements.txt --allow-all-external

>git init
>git add server.py requirements.txt
>git commit -m "initial"
>git log

>heroku create
  Creating quiet-stream-2985... done, stack is cedar
  http://quiet-stream-2985.herokuapp.com/ | git@heroku.com:quiet-stream-2985.git
  Git remote heroku added
>cat .git\config
>git push heroku master
  -----> Installing runtime (python-2.7.8)

So this used the wrong python runtime, aiohttp needs 3.4.1+:

>echo python-3.4.1> runtime.txt
>git add runtime.txt
>git commit -m "need python 3.4"
>git push heroku master
  http://quiet-stream-2985.herokuapp.com/ deployed to Heroku

So now it succeeded, or at least didn't spew out Python 2 vs 3 mismatch errors. But when I browse http://quiet-stream-2985.herokuapp.com/ I get an error.

>heroku ps:scale web=1
  Scaling dynos... failed
   !    No such process type web defined in Procfile.

Oops. Actually how does Heroku know it has to exec 'python server.py'? Atm it cannot. Looking at https://github.com/heroku/python-getting-started/blob/master/Procfile: this starts gunicorn. So:

>echo "web: python server.py"> Procfile; notepad Procfile
>git add Procfile
>git commit -m "Procfile was missing"
>git push heroku master

Now retry starting the web server:

>heroku ps:scale web=1
  Scaling dynos... done, now running web at 1:1X.

So thats better. I still cannot browse the page now though, and curl times out:

>curl --max-time 3 http://quiet-stream-2985.herokuapp.com/config

>heroku logs --tail
  Slug compilation failed: failed to compile Python app
  Release v4 created by kjell.schubert@mmodal.com
  ...

Not very obvious to me whats wrong.

>heroku ps
  === web (1X): `python server.py`
  web.1: crashed 2014/10/21 16:28:17 (~ 4m ago)

Running app locally:

>foreman start web

This allows me to access http://localhost:8480/ just fine. Actually which port should I server the webs on? 80? My server listens on 8480, but Heroku cannot easily know that (unless they scan the process for open ports, which would be silly). Judging from http://stackoverflow.com/questions/13714205/deploying-flask-app-to-heroku I need to read a PORT env var?

>curl --max-time 3 http://quiet-stream-2985.herokuapp.com/config
  {"testProperty2": true, "testProperty": "noMachineSpecified", "testProperty3": 123}

Nice! Works! Well that was pretty straightforward :) Now to not waste a VM at heroku.com let's stop the server:

>heroku ps:scale web=0
>curl --max-time 3 http://quiet-stream-2985.herokuapp.com/config
   yields error, good