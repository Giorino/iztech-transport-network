package com.izmir.transportation;

import java.awt.Color;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.geometry.jts.JTSFactoryFinder;
import org.geotools.map.FeatureLayer;
import org.geotools.map.MapContent;
import org.geotools.referencing.crs.DefaultGeographicCRS;
import org.geotools.styling.SLD;
import org.geotools.styling.Style;
import org.geotools.swing.JMapFrame;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.LinearRing;
import org.locationtech.jts.geom.Point;
import org.locationtech.jts.geom.Polygon;
import org.opengis.feature.simple.SimpleFeatureType;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * A class for generating and visualizing random points within the Izmir bay area.
 * Points are generated based on population density, ensuring a realistic distribution
 * of vertices for the transportation network analysis.
 * 
 * @author yagizugurveren
 */
public class IzmirBayGraph {
    private static final GeometryFactory geometryFactory = JTSFactoryFinder.getGeometryFactory();
    private static Polygon izmirBoundary;

    private static final Map<Point, Double> POPULATION_CENTERS = new HashMap<>();

    static {
        loadIzmirBoundary();
        loadPopulationCenters();
    }

    public static void main(String[] args) {
        try {
            // Generate random points
            System.out.println("Generating random vertices...");
            List<Point> points = generateRandomPoints(2500, 0.01);

            // Save points to CSV
            System.out.println("Saving points to CSV...");
            savePointsToCSV(points, "random_izmir_points.csv");

            // Visualize points
            System.out.println("Visualizing points...");
            visualizePoints(points);

            System.out.println("Done! Points have been generated and saved.");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Loads the Izmir boundary from a GeoJSON file in the resources.
     * The boundary is used to constrain the generated points within Izmir's limits.
     */
    private static void loadIzmirBoundary() {
        try {
            try (InputStream is = IzmirBayGraph.class.getResourceAsStream("/izmir.json")) {
                if (is == null) {
                    throw new IOException("Could not find izmir.json in resources");
                }
                String jsonContent = new String(is.readAllBytes(), StandardCharsets.UTF_8);
                ObjectMapper mapper = new ObjectMapper();
                JsonNode root = mapper.readTree(jsonContent);
                JsonNode coordinates = root.path("geometry").path("coordinates").get(0);

                Coordinate[] coords = new Coordinate[coordinates.size()];
                for (int i = 0; i < coordinates.size(); i++) {
                    JsonNode coord = coordinates.get(i);
                    double lon = coord.get(0).asDouble();
                    double lat = coord.get(1).asDouble();
                    coords[i] = new Coordinate(lon, lat);
                }
                // Close the ring by adding the first coordinate again
                if (!coords[0].equals(coords[coords.length - 1])) {
                    Coordinate[] closedCoords = new Coordinate[coords.length + 1];
                    System.arraycopy(coords, 0, closedCoords, 0, coords.length);
                    closedCoords[coords.length] = coords[0];
                    coords = closedCoords;
                }

                LinearRing ring = geometryFactory.createLinearRing(coords);
                izmirBoundary = geometryFactory.createPolygon(ring);
            }
        } catch (IOException e) {
            System.err.println("Error reading Izmir boundary: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Loads population centers and their weights from a CSV file in the resources.
     * The weights are normalized based on the population of each center.
     */
    private static void loadPopulationCenters() {
        try (InputStream is = IzmirBayGraph.class.getResourceAsStream("/neighborhood_population.csv")) {
            if (is == null) {
                throw new IOException("Could not find neighborhood_population.csv in resources");
            }
            Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8);
            CSVParser csvParser = CSVFormat.DEFAULT
                    .withFirstRecordAsHeader()
                    .parse(reader);

            double maxPopulation = 0;
            // First pass to find max population for normalization
            for (CSVRecord record : csvParser) {
                double population = Double.parseDouble(record.get("Population"));
                if (population > maxPopulation) {
                    maxPopulation = population;
                }
            }

            // Reset the reader and parser
            is.close();
            InputStream secondIs = IzmirBayGraph.class.getResourceAsStream("/neighborhood_population.csv");
            if (secondIs == null) {
                throw new IOException("Could not find neighborhood_population.csv in resources");
            }
            Reader secondReader = new InputStreamReader(secondIs, StandardCharsets.UTF_8);
            csvParser = CSVFormat.DEFAULT
                    .withFirstRecordAsHeader()
                    .parse(secondReader);

            // Second pass to add normalized population centers
            for (CSVRecord record : csvParser) {
                double lon = Double.parseDouble(record.get("Longitude"));
                double lat = Double.parseDouble(record.get("Latitude"));
                double population = Double.parseDouble(record.get("Population"));

                // Normalize population to get weight between 0 and 1
                double weight = population / maxPopulation;

                Point point = geometryFactory.createPoint(new Coordinate(lon, lat));
                POPULATION_CENTERS.put(point, weight);
            }
        } catch (IOException e) {
            System.err.println("Error reading neighborhood population data: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Generates a specified number of random points within the Izmir boundary.
     * Points are generated with respect to population centers and their weights.
     *
     * @param numPoints The number of points to generate
     * @param standardDeviation The standard deviation for the Gaussian distribution around population centers
     * @return List of generated points
     */
    private static List<Point> generateRandomPoints(int numPoints, double standardDeviation) {
        List<Point> points = new ArrayList<>();
        Random random = new Random();

        // Add hardcoded IYTE point
        Point iytePoint = geometryFactory.createPoint(new Coordinate(26.643221256222105, 38.319147501994145));
        points.add(iytePoint);

        while (points.size() < numPoints) {
            // generate points relative to population centers
            Point point = generatePointWithSpread(random, standardDeviation);

            // Only add the point if it's within the Izmir boundary
            if (izmirBoundary.contains(point)) {
                points.add(point);
            }
        }

        return points;
    }

    /**
     * Generates a single point with Gaussian spread around a randomly selected population center.
     * The selection of the center is weighted by population.
     *
     * @param random Random number generator
     * @param standardDeviation Standard deviation for the Gaussian distribution
     * @return A new point generated around a population center
     */
    private static Point generatePointWithSpread(Random random, double standardDeviation) {
        // Select a random population center based on weights
        double totalWeight = POPULATION_CENTERS.values().stream().mapToDouble(Double::doubleValue).sum();
        double r = random.nextDouble() * totalWeight;

        Point center = null;
        for (Map.Entry<Point, Double> entry : POPULATION_CENTERS.entrySet()) {
            r -= entry.getValue();
            if (r <= 0) {
                center = entry.getKey();
                break;
            }
        }

        if (center == null) {
            center = POPULATION_CENTERS.keySet().iterator().next();
        }

        Point candidate;
        do {
            double spreadMultiplier = 1.0 + random.nextGaussian(); 
            double effectiveSpread = standardDeviation * spreadMultiplier;
            
            double lonOffset = random.nextGaussian() * effectiveSpread;
            double latOffset = random.nextGaussian() * effectiveSpread;
            
            double lon = center.getX() + lonOffset;
            double lat = center.getY() + latOffset;
            
            candidate = geometryFactory.createPoint(new Coordinate(lon, lat));
        } while (!izmirBoundary.contains(candidate));

        return candidate;
    }

    /**
     * Saves the generated points to a CSV file.
     *
     * @param points List of points to save
     * @param filename Name of the output CSV file
     * @throws IOException If there is an error writing to the file
     */
    private static void savePointsToCSV(List<Point> points, String filename) throws IOException {
        try (CSVPrinter printer = new CSVPrinter(
                new FileWriter(filename, StandardCharsets.UTF_8),
                CSVFormat.DEFAULT.withHeader("longitude", "latitude"))) {
            for (Point point : points) {
                printer.printRecord(point.getX(), point.getY());
            }
        }
    }

    /**
     * Visualizes the generated points and the Izmir boundary on a map.
     * Creates a GUI window showing the points overlaid on the boundary.
     *
     * @param points List of points to visualize
     * @throws Exception If there is an error creating or showing the map
     */
    private static void visualizePoints(List<Point> points) throws Exception {
        // Create feature types
        SimpleFeatureTypeBuilder pointBuilder = new SimpleFeatureTypeBuilder();
        pointBuilder.setName("Points");
        pointBuilder.setCRS(DefaultGeographicCRS.WGS84);
        pointBuilder.add("geometry", Point.class);
        SimpleFeatureType pointType = pointBuilder.buildFeatureType();

        SimpleFeatureTypeBuilder boundaryBuilder = new SimpleFeatureTypeBuilder();
        boundaryBuilder.setName("Boundary");
        boundaryBuilder.setCRS(DefaultGeographicCRS.WGS84);
        boundaryBuilder.add("geometry", Polygon.class);
        SimpleFeatureType boundaryType = boundaryBuilder.buildFeatureType();

        // Create features collections
        DefaultFeatureCollection pointCollection = new DefaultFeatureCollection();
        DefaultFeatureCollection boundaryCollection = new DefaultFeatureCollection();

        // Add points
        SimpleFeatureBuilder pointFeatureBuilder = new SimpleFeatureBuilder(pointType);
        for (Point point : points) {
            pointFeatureBuilder.add(point);
            pointCollection.add(pointFeatureBuilder.buildFeature(null));
        }

        // Add boundary
        SimpleFeatureBuilder boundaryFeatureBuilder = new SimpleFeatureBuilder(boundaryType);
        boundaryFeatureBuilder.add(izmirBoundary);
        boundaryCollection.add(boundaryFeatureBuilder.buildFeature(null));

        // Create styles
        Style pointStyle = SLD.createPointStyle("circle", Color.BLUE, Color.BLUE, 0.5f, 5);
        Style boundaryStyle = SLD.createSimpleStyle(boundaryType);

        // Create map
        MapContent map = new MapContent();
        map.setTitle("Izmir Bay Random Points");
        map.addLayer(new FeatureLayer(boundaryCollection, boundaryStyle));
        map.addLayer(new FeatureLayer(pointCollection, pointStyle));

        // Show map
        JMapFrame.showMap(map);
    }
} 