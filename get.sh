IP_ADD="172.16.9.178"
USERNAME="wuli"
PASSWORD="111"
ftp -v -n $IP_ADD<<- EOF
user $USERNAME $PASSWORD
binary
ls
#cd $REMOTE_PATH
get $1
byeEOF
