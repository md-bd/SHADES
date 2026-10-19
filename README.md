<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<br />
<div align="center">

  <h3 align="center"><i>SHADES</i></h3>

  <p align="center">
    <i>SHADES</i>: Statistics of Haplotype frequencies And Differences for Evolutionary Signatures <br />
    <br />
    <a href="https://github.com/md-bd/SHADES/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/md-bd/SHADES/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
        <a href="#usage">Usage</a>
        <ul>
            <li><a href="#statistics-generation">Statistics generation</a></li>
            <li><a href="#model-training-and-testing">Model Training and Testing</a></li>
            <li><a href="#empirical-scan">Empirical scan</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

$\textit{SHADES}$ is a transfer learning framework to distinguish among neutrality, selective sweeps, balancing selection, and adaptive introgression.  


<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->



Built With

[![Python][Python.com]][Python-url]

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Python version `3.8.18` or above is necessary to use this software. Run the following commands to ensure you have the required version.
* check python version
  ```sh
  python --version
  ```

### Installation
 
1. Clone the repo
   ```sh
   git clone https://github.com/md-bd/SHADES.git
   ```
2. Go to project directory
    ```sh
    cd SHADES
    ```
3. Install python packages
   ```sh
   pip install -r requirements.txt
   ```


**NOTE**: If you prefer using docker and want to install docker image, follow the instructions here: TODO<!-- TODO -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Our approach creates statistics from population genomic data. Then we train and test CNN model on genomic data.  

### Statistics generation
We simulated CEU demography data in SLiM3. The file format is `ms` for the simulation. We also do empirical scan and the file format for it is `vcf`. 
Both format data can be parsed and used for statistics generation using our scripts.  
#### ms file 

```Python
# parse image 1 and 3:
python src/parse_data.py -c configs/parse_w11.yaml 
# parse image 2:
python src/dist_from_freq.py -c configs/parse_w11.yaml
```

#### vcf file

```Python
python src/parse_chromosome.py -c configs/parse_chromosome_w11.yaml
```

You have to generate statistics for window 11, 21, and 51 separately for simulation and empirical data. 

### Model Training and Testing
$\textit{SHADES}$ model training process: 
1. Train M (M1 or M2) architecture model for window 11, 21, and 51 data. 
2. Use trained M model for each window to train $\textit{SHADES}[M]$ model. We will keep the learned M model parameters frozen.

#### $\textit{SHADES}[M_1]$
M1 model training with window 11 example:
```Python
python src/cnn_train.py -c configs/train_w11_m1.yaml
```
Similarly we need to train w21 and w51 M1 model. When we have all windows model ready, we are ready to train the stacked model ($\textit{SHADES}[M_1]$).  

```Python
# SHADES[M1]
python src/cnn_train.py -c configs/train_sm1.yaml
```


#### $\textit{SHADES}[M_2]$
M2 model training with window 11 example:
```Python
python src/cnn_train_inception_resnet_v2.py -c configs/train_w11_m2.yaml
```
Similarly we need to train w21 and w51 M2 model. When we have all windows model ready, we are ready to train the stacked model ($\textit{SHADES}[M_2]$).  

```Python
# SHADES[M2]
python src/cnn_train_inc_resv2_stacking.py -c configs/train_sm2.yaml
```

### Empirical scan

With trained models, we can scan the human genome and get predictions on evolutionary classes.  
```Python
# SHADES[M1]
python src/cnn_infer.py -c configs/infer_chromosome.yaml
```

```Python
# SHADES[M2]
python src/cnn_infer_transfer_learning.py -c configs/infer_chromosome.yaml
```

### How to
There are other things our method is capable of. Here are some examples of getting these interesting things done.
#### Run using Unphased Dataset

Parse Data: 
```Python
python src/parse_data_unphased.py -c configs/parse_unphased_w11.yaml
```

Train and test is same as phased data train and test process and code.  



<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Mohammad Khan - mkhan2024@fau.edu

Project Link: [https://github.com/md-bd/SHADES](https://github.com/md-bd/SHADES)

Profile page: [https://md-bd.github.io/](https://md-bd.github.io/) 


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [readme template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/md-bd/SHADES.svg?style=for-the-badge
[contributors-url]: https://github.com/md-bd/SHADES/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/md-bd/SHADES.svg?style=for-the-badge
[forks-url]: https://github.com/md-bd/SHADES/network/members
[stars-shield]: https://img.shields.io/github/stars/md-bd/SHADES.svg?style=for-the-badge
[stars-url]: https://github.com/md-bd/SHADES/stargazers
[issues-shield]: https://img.shields.io/github/issues/md-bd/SHADES.svg?style=for-the-badge
[issues-url]: https://github.com/md-bd/SHADES/issues
[license-shield]: https://img.shields.io/github/license/md-bd/SHADES.svg?style=for-the-badge
[license-url]: https://github.com/md-bd/SHADES/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/mohammad-khan-buet
[product-screenshot]: images/screenshot.png
<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
[Python.com]: https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff
[Python-url]: https://www.python.org/