step=$1
echo $step

# Alternatively, run a subshell using the sh command
sh <<EOF
    echo "shell 1"
    bash test_env.sh
EOF

# echo "main shell"

sh <<EOF
    echo "shell 2"
    bash test.sh $step
EOF