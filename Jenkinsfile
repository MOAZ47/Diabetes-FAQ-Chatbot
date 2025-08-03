pipeline {
    agent any

    environment {
        MISTRAL_API_KEY = credentials('MISTRAL_API_KEY')
        TAVILY_API_KEY = credentials('tavily-api')
        HF_TOKEN = credentials('hf-token')
        PINECONE_API_KEY = credentials('PINECONE_API_KEY')
        WANDB_API_KEY = credentials('WANDB_API_KEY')
    }

    stages {

        stage ('Checkout Github') {
            steps {
                script {
                    if (fileExists('.git')) {
                        echo "Local repo exists. Skipping git checkout."
                    } else {
                        echo "Cloning repository from GitHub..."
                        git url: 'https://github.com/MOAZ47/Diabetes-FAQ-Chatbot'
                    }
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    def imageExists = sh (
                      script: "docker images -q moaz47/diabetes-faq-app:latest",
                      returnStdout: true
                    ).trim()

                    if (imageExists) {
                      echo "Docker image already exists: Skipping build."
                    } else {
                      echo "Image not found: Building Docker image..."
                      // sh "docker build -t moaz47/diabetes-faq-app:latest ."
                    }
                }
            }
        }

        stage('Test') {
            parallel {
                stage('LLM Evaluation') {
                  steps {
                    echo 'Running TruLens Evaluation...'
                    sh 'python evaluation.py'
                  }
                }

                stage('Weights & Biases Evaluation') {
                  steps {
                    echo 'Running Weights and Biases Eval'
                    sh 'python evaluation_wnb.py'
                  }
                }
            }
        }

        stage('Deploy') {
            steps {
              echo 'Image already exists on Docker Hub, skipping deployment.'

              
              withCredentials([usernamePassword(
                credentialsId: 'dockerhub-creds',
                usernameVariable: 'DOCKER_USER',
                passwordVariable: 'DOCKER_PASS'
              )]) {
                // replace echo with sh ''' 
                echo """
                  echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                  docker tag moaz47/diabetes-faq-app:latest moaz47/diabetes-faq-app:1.1
                  docker push moaz47/diabetes-faq-app:1.1
                """
              }
              
            }
        }
    }
}
