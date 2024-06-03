# Define targets for running pre-commit and cz-commit

# pre-commit: Run pre-commit checks on all files
# This target will run the pre-commit hooks on all files in the repo.
# This is useful for running checks before committing.
pre-commit:
	# Run pre-commit hooks on all files
	pre-commit run --all-files

# cz-commit: Run commitizen commit
# This target will run the cz-commit command to create a new commit.
# This is useful for creating commits that follow the commitzen conventions.
cz-commit:
	# Run cz-commit command
	cz commit

# commit: Run pre-commit and cz-commit
# This target will run the pre-commit checks and then run the cz-commit command.
# This is useful for the common use case of creating a new commit.
commit:
	# Run pre-commit checks on all files
	$(MAKE) pre-commit
	# Run cz-commit command to create a new commit
	$(MAKE) cz-commit
