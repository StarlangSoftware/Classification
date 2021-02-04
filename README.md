For Developers
============

You can also see [Python](https://github.com/starlangsoftware/Classification-Py), [Cython](https://github.com/starlangsoftware/Classification-Cy), [C++](https://github.com/starlangsoftware/Classification-CPP), or [C#](https://github.com/starlangsoftware/Classification-CS) repository.

## Requirements

* [Java Development Kit 8 or higher](#java), Open JDK or Oracle JDK
* [Maven](#maven)
* [Git](#git)

### Java 

To check if you have a compatible version of Java installed, use the following command:

    java -version
    
If you don't have a compatible version, you can download either [Oracle JDK](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html) or [OpenJDK](https://openjdk.java.net/install/)    

### Maven
To check if you have Maven installed, use the following command:

    mvn --version
    
To install Maven, you can follow the instructions [here](https://maven.apache.org/install.html).      

### Git

Install the [latest version of Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

## Download Code

In order to work on code, create a fork from GitHub page. 
Use Git for cloning the code to your local or below line for Ubuntu:

	git clone <your-fork-git-link>

A directory called Classification will be created. Or you can use below link for exploring the code:

	git clone https://github.com/olcaytaner/Classification.git

## Open project with IntelliJ IDEA

Steps for opening the cloned project:

* Start IDE
* Select **File | Open** from main menu
* Choose `Classification/pom.xml` file
* Select open as project option
* Couple of seconds, dependencies with Maven will be downloaded. 


## Compile

**From IDE**

After being done with the downloading and Maven indexing, select **Build Project** option from **Build** menu. After compilation process, user can run Classification.

**From Console**

Go to `Classification` directory and compile with 

     mvn compile 

## Generating jar files

**From IDE**

Use `package` of 'Lifecycle' from maven window on the right and from `Classification` root module.

**From Console**

Use below line to generate jar file:

     mvn install

## Maven Usage

        <dependency>
            <groupId>io.github.starlangsoftware</groupId>
            <artifactId>Classification</artifactId>
            <version>1.0.1</version>
        </dependency>

Detailed Description
============

+ [Classification Algorithms](#classification-algorithms)
+ [Sampling Strategies](#sampling-strategies)
+ [Feature Selection](#feature-selection)
+ [Statistical Tests](#statistical-tests)

## Classification Algorithms

Algoritmaları eğitmek için

	void train(InstanceList trainSet, Parameter parameters)

Eğitilen modeli bir veri örneği üstünde sınamak için

	String predict(Instance instance)

Karar ağacı algoritması C45 sınıfında

Bagging algoritması Bagging sınıfında

Derin öğrenme algoritması DeepNetwork sınıfında

KMeans algoritması KMeans sınıfında

Doğrusal ve doğrusal olmayan çok katmanlı perceptron LinearPerceptron ve 
MultiLayerPerceptron sınıflarında

Naive Bayes algoritması NaiveBayes sınıfında

K en yakın komşu algoritması Knn sınıfında

Doğrusal kesme analizi algoritması Lda sınıfında

İkinci derece kesme analizi algoritması Qda sınıfında

Destek vektör makineleri algoritması Svm sınıfında

RandomForest ağaç tabanlı ensemble algoritması RandomForest sınıfında

Basit dummy ve rasgele sınıflandırıcı gibi temel iki sınıflandırıcı Dummy ve 
RandomClassifier sınıflarında

## Sampling Strategies

K katlı çapraz geçerleme deneyi yapmak için KFoldRun, KFoldRunSeparateTest, 
StratifiedKFoldRun, StratifiedKFoldRunSeparateTest

M tane K katlı çapraz geçerleme deneyi yapmak için MxKFoldRun, MxKFoldRunSeparateTest,
StratifiedMxKFoldRun, StratifiedMxKFoldRunSeparateTest

Bootstrap tipi deney yapmak için BootstrapRun

## Feature Selection

Pca tabanlı boyut azaltma için Pca sınıfı

Discrete değişkenleri Continuous değişkenlere çevirmek için DiscreteToContinuous sınıfı

Discrete değişkenleri binary değişkenlere değiştirmek için LaryToBinary sınıfı

## Statistical Tests

İstatistiksel testler için Combined5x2F, Combined5x2t, Paired5x2t, Pairedt, Sign sınıfları
