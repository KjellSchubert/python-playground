(Personal Python cheat sheet)

To install deps same way as travis-ci and heroku, equivalent to npm install packacke.json:

      pip install virtualenv
      virtualenv venv
      pip install -r requirements.txt --upgrade

To run all unit tests: 

      python -m unittest discover -s tests
    
P.S.: switched to nosetest since this seems more comfortable to use:

      pip install nose
      nosetests

PyDev: requires Java 1.7 in Eclipse Luna, silently fails with 1.6. Can run nosetests in graphical debugger.

To run a single unit test:

      python -m unittest tests.test_koa.KoaAppTestCase.test_koa_router_http_post
      nosetests tests/test_admin_app.py:AdminAppTestCase.test_GET_conditionPropertyValues

To stop at a breakpoint in nose just add -s 
(see [here](http://stackoverflow.com/questions/4950637/setting-breakpoints-with-nosetests-pdb-option)). When using 'python -m unittest' no additional options are needed to hit break points.

      nosetests -s tests/test_admin_app.py:AdminAppTestCase.test_GET_conditionPropertyValues
      
PDB cmd docu is [here](https://docs.python.org/2/library/pdb.html#debugger-commands)

To get cov for unit tests (see also .travis.yml for how to run):

      pip install coverage
      coverage run -m unittest discover -s tests
      coverage report -m

To replace mixedCase with Python-preferred case:

      sed -i -e :loop -re "s/(^|[^A-Za-z_])([a-z0-9_]+)([A-Z])([A-Za-z0-9_]*)([^A-Za-z0-9_]|$)/\1\2_\l\3\4\5/" -e "t loop" foo.py

C10k - capable Python servers are:

* aiohttp https://pypi.python.org/pypi/aiohttp/0.9.1 based on 3.4 asyncio packages
* tornado
* twisted: these guys don't seem to care about python 3.x, it's 2.x only atm
* flask? I don't think flask is C10k capable, it seems to be sync-IO
* django? there's an alpha for this at https://github.com/aaugustin/django-c10k-demo