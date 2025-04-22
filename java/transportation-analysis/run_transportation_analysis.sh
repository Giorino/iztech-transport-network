#!/bin/bash

# Script to run the Transportation Analysis application

echo "Starting Transportation Analysis..."

# Build the project with assembly plugin
mvn clean package -DskipTests

# Run the analysis with sufficient memory using the executable JAR
java -Xmx2g -jar target/transportation-analysis-1.0-SNAPSHOT-jar-with-dependencies.jar

echo "Analysis complete!" 