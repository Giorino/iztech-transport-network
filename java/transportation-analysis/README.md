# Analyzing Transportation Network of Izmir Institute of Technology using Graph Theory and Algorithms

This project analyzes the transportation network around Izmir Institute of Technology (IYTE) using graph theory and algorithms. It creates a weighted graph representation of the road network, with a focus on the IYTE campus and surrounding areas.

## Features

- Generation of random vertices based on population density
- Integration with OpenStreetMap data for real road network information
- Visualization of the transportation network using GeoTools
- Graph-based analysis of transportation paths
- Distance-weighted connections between points

## Prerequisites

- Java 8 or higher
- Maven
- Internet connection (for downloading OpenStreetMap data)

## Building the Project

To build the project, run:

```bash
mvn clean install
```

## Running the Application

After building, you can run the application using:

```bash
mvn exec:java -Dexec.mainClass="com.izmir.App"
```

## Project Structure

- `src/main/java/com/izmir/`
  - `App.java` - Main application entry point
  - `transportation/` - Core transportation analysis classes
    - `TransportationGraph.java` - Graph representation of the network
    - `CreateRoadNetwork.java` - Road network creation from OSM data
    - `IzmirBayGraph.java` - Random vertex generation
    - `OSMUtils.java` - OpenStreetMap data utilities

## Output

The application generates several output files:
- `random_izmir_points.csv` - Generated points data
- `izmir_network_nodes.csv` - Network node locations
- Visual map displays showing:
  - Base road network
  - Generated points
  - Connected paths
