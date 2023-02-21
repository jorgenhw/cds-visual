# Add a new remove upstram repo
git remote add upstream 

# Sync fork
git fetch upstream
git checkout main
git merge upstream/main

#generate requirements file automatically
%pip freeze > requirements.txt