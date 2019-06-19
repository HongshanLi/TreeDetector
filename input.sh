echo Hello, who are you?

read -t 5 varname

if [ $varname=no ]; then
        echo not quiting
        exit 1
fi

echo Nice to meet you
