# had to remake makefile so its windows compatible 
# Define your virtual environment and Flask app
VENV = venv
FLASK_APP = app.py

# Determine the correct pip and flask paths based on the operating system
ifeq ($(OS),Windows_NT)
	PIP = .\$(VENV)\Scripts\pip
	FLASK = .\$(VENV)\Scripts\flask
	ENV_SET = set
else
	PIP = ./$(VENV)/bin/pip
	FLASK = ./$(VENV)/bin/flask
	ENV_SET = export
endif

# Install dependencies
install:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

# Run the Flask application
run:
	$(ENV_SET) FLASK_APP=$(FLASK_APP) && $(ENV_SET) FLASK_ENV=development && $(FLASK) run --port 3000

# Clean up virtual environment
clean:
	rm -rf $(VENV)

# Reinstall all dependencies
reinstall: clean install
