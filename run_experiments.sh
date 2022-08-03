n=1
a=dqn.json
t=7200
for (( i=1; i<=n; i++ ))
do
    rm -rf saved-model
    python3 wh_create_model.py -a $a
    python3 wh_learn_enc.py -s -t $t
done