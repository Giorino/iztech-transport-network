package com.izmir.transportation;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.Envelope;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.LineString;
import org.locationtech.jts.geom.Point;

public class OSMDataCache {
    private static final Logger LOGGER = Logger.getLogger(OSMDataCache.class.getName());
    private static final String CACHE_DIR = "osm_cache";
    private static final String NODES_FILE = CACHE_DIR + "/nodes.csv";
    private static final String WAYS_FILE = CACHE_DIR + "/ways.csv";
    private static final GeometryFactory geometryFactory = new GeometryFactory();

    public static Map<String, Object> getOrDownloadData(Envelope bbox) throws Exception {
        if (isCacheValid()) {
            LOGGER.info("Loading data from cache...");
            return loadFromCache();
        } else {
            LOGGER.info("Cache not found or invalid. Downloading fresh data...");
            Map<String, Object> data = OSMUtils.downloadRoadNetwork(bbox);
            saveToCache(data);
            return data;
        }
    }

    private static boolean isCacheValid() {
        File nodesFile = new File(NODES_FILE);
        File waysFile = new File(WAYS_FILE);
        
        // Check if cache files exist and are not older than 7 days
        if (!nodesFile.exists() || !waysFile.exists()) {
            return false;
        }
        
        long sevenDaysInMillis = 7 * 24 * 60 * 60 * 1000;
        long now = System.currentTimeMillis();
        return (now - nodesFile.lastModified() < sevenDaysInMillis) &&
               (now - waysFile.lastModified() < sevenDaysInMillis);
    }

    private static void saveToCache(Map<String, Object> data) throws IOException {
        File cacheDir = new File(CACHE_DIR);
        if (!cacheDir.exists()) {
            cacheDir.mkdirs();
        }

        // Save nodes
        LOGGER.info("Saving nodes to cache...");
        @SuppressWarnings("unchecked")
        Map<String, Point> nodes = (Map<String, Point>) data.get("nodes");
        try (CSVPrinter printer = new CSVPrinter(
                new FileWriter(NODES_FILE, StandardCharsets.UTF_8),
                CSVFormat.DEFAULT.withHeader("id", "longitude", "latitude"))) {
            for (Map.Entry<String, Point> entry : nodes.entrySet()) {
                printer.printRecord(
                    entry.getKey(),
                    entry.getValue().getX(),
                    entry.getValue().getY()
                );
            }
        }

        // Save ways
        LOGGER.info("Saving ways to cache...");
        @SuppressWarnings("unchecked")
        List<LineString> ways = (List<LineString>) data.get("ways");
        try (CSVPrinter printer = new CSVPrinter(
                new FileWriter(WAYS_FILE, StandardCharsets.UTF_8),
                CSVFormat.DEFAULT.withHeader("way_id", "coordinates"))) {
            for (int i = 0; i < ways.size(); i++) {
                LineString way = ways.get(i);
                StringBuilder coords = new StringBuilder();
                for (Coordinate coord : way.getCoordinates()) {
                    if (coords.length() > 0) coords.append(";");
                    coords.append(coord.x).append(",").append(coord.y);
                }
                printer.printRecord(i, coords.toString());
            }
        }
        LOGGER.info("Cache saved successfully");
    }

    private static Map<String, Object> loadFromCache() throws IOException {
        Map<String, Object> result = new HashMap<>();
        
        // Load nodes
        LOGGER.info("Loading nodes from cache...");
        Map<String, Point> nodes = new HashMap<>();
        try (CSVParser parser = new CSVParser(
                new FileReader(NODES_FILE, StandardCharsets.UTF_8),
                CSVFormat.DEFAULT.withFirstRecordAsHeader())) {
            for (CSVRecord record : parser) {
                String id = record.get("id");
                double lon = Double.parseDouble(record.get("longitude"));
                double lat = Double.parseDouble(record.get("latitude"));
                nodes.put(id, geometryFactory.createPoint(new Coordinate(lon, lat)));
            }
        }
        result.put("nodes", nodes);

        // Load ways
        LOGGER.info("Loading ways from cache...");
        List<LineString> ways = new ArrayList<>();
        try (CSVParser parser = new CSVParser(
                new FileReader(WAYS_FILE, StandardCharsets.UTF_8),
                CSVFormat.DEFAULT.withFirstRecordAsHeader())) {
            for (CSVRecord record : parser) {
                String[] coordPairs = record.get("coordinates").split(";");
                Coordinate[] coordinates = new Coordinate[coordPairs.length];
                for (int i = 0; i < coordPairs.length; i++) {
                    String[] lonLat = coordPairs[i].split(",");
                    coordinates[i] = new Coordinate(
                        Double.parseDouble(lonLat[0]),
                        Double.parseDouble(lonLat[1])
                    );
                }
                ways.add(geometryFactory.createLineString(coordinates));
            }
        }
        result.put("ways", ways);

        LOGGER.info(String.format("Loaded %d nodes and %d ways from cache", 
            nodes.size(), ways.size()));
        return result;
    }
} 