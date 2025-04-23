-- AppleScript to run Docker container through the shell script
tell application "Terminal"
	-- Activate Terminal
	activate
	
	-- Create a new terminal window
	do script ""
	
	-- Navigate to project directory and run the script
	do script "cd /Users/wvansickle/gtm-scrape-1 && ./run_docker.sh" in front window
end tell 