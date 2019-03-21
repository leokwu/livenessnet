IP_ADD="172.16.9.178"
USERNAME="wuli"
PASSWORD="11"
ftp -v -n $IP_ADD<<- EOF
user $USERNAME $PASSWORD
binary
#cd $REMOTE_PATH
put $1
byeEOF
