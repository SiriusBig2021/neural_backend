## SIRIUS 2021 NN Backend

### Installation 

```
sudo chmod +x ./install_linux.sh
./install_linux.sh
```


### Useful info



#### Microsoft Azure server
Используем для хранения данных, обучения нейросетей

```
Как зайти по SSH:
ssh admsirius@51.136.7.135
## Gfhjkm4adm1!

Открыть доступ к своему ПК по ssh в локальной сети:
sudo apt install openssh-server

Как перекидывать данные по ssh:
rsync -avp --progress %PATH TO FILE/FOLDER% admsirius@51.136.7.135 %PATH TO DESTINATION%
```



#### Terminal commands
При вводе пользуемся активно Tab для автодополнения ввода


Navigation

```
1. cd /home/USER/projects/
2. cd .. 
3. ls
```

Virtual environment

```
1. mkvirtualenv -p python3.8 vEnvName
2. workon vEnvName
3. deactivate
4. rmvirtualenv vEnvName
```

#### Links

- Введение в ML и CV
  
    [Презентация](https://docs.google.com/presentation/d/1j6QyKOEXyHoeq9TgjJwEpqYx3UGctZfc2N12xPGOhVI/edit?usp=sharing)

- Обучение нейросети

    [Презентация](https://docs.google.com/presentation/d/1eVhWJDOH6fJ6RWuF6KCLkSUmEpP40s4R2sInz3kJXl0/edit?usp=sharing)

    [Видосы по BackProp](https://www.youtube.com/watch?v=aircAruvnKk) Русские субтитры есть.

- Аугментации
  
    [Imgaug](https://github.com/aleju/imgaug)
    

