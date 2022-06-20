An example command to compute the Hessian max eigenvalue and trace:
```
python compute_hessian.py --model_dir ../logs/<folder_of_run>/server.pt
```
If you would also like the landscape plot, add the ```--plot``` flag. Note that plotting can take a considerable amount of time. For cross-client information, you will need to conduct the training with the ```--save_client``` flag to save all client model weights. For example:
```
python main.py --method fedavg <all_other_args> --save_client
```
Then, compute the second-order information for all models:
```
python compute_hessian.py --model_dir ../logs/<folder_of_run>/server.pt
python compute_hessian.py --model_dir ../logs/<folder_of_run>/client_0.pt
python compute_hessian.py --model_dir ../logs/<folder_of_run>/client_1.pt
...
```
Finally, run the follwing command:
```
python parse_logs.py --log_dir ../logs/<folder_of_run>/hessian
```
