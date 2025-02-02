package com.izmir.transportation;

import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.Envelope;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.LineString;
import org.locationtech.jts.geom.Point;
import org.locationtech.jts.io.WKTReader;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

public class OSMUtils {
    private static final Logger LOGGER = Logger.getLogger(OSMUtils.class.getName());
    private static final String OVERPASS_API_URL = "https://overpass-api.de/api/interpreter";
    private static final GeometryFactory geometryFactory = new GeometryFactory();
    private static final WKTReader wktReader = new WKTReader();

    public static Map<String, Object> downloadRoadNetwork(Envelope bbox) throws Exception {
        LOGGER.info("Starting road network download for bbox: " + bbox);
        
        // Create Overpass QL query
        String query = String.format(
                "[out:xml][timeout:25];" +
                "way[\"highway\"](%f,%f,%f,%f);" +
                "(._;>;);" +
                "out body;",
                bbox.getMinY(), bbox.getMinX(), bbox.getMaxY(), bbox.getMaxX()
        );
        LOGGER.info("Generated Overpass query: " + query);

        // Download data from Overpass API
        LOGGER.info("Connecting to Overpass API...");
        URL url = new URL(OVERPASS_API_URL);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setDoOutput(true);

        String postData = "data=" + URLEncoder.encode(query, StandardCharsets.UTF_8);
        LOGGER.info("Sending request to Overpass API...");
        conn.getOutputStream().write(postData.getBytes(StandardCharsets.UTF_8));

        // Parse response
        LOGGER.info("Parsing XML response...");
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document doc = builder.parse(conn.getInputStream());

        // Extract nodes
        LOGGER.info("Extracting nodes from response...");
        Map<String, Point> nodes = new HashMap<>();
        NodeList nodeElements = doc.getElementsByTagName("node");
        int totalNodes = nodeElements.getLength();
        LOGGER.info("Found " + totalNodes + " nodes");
        
        for (int i = 0; i < nodeElements.getLength(); i++) {
            if (i % 1000 == 0) {
                LOGGER.info(String.format("Processing nodes: %.1f%% complete", (i * 100.0) / totalNodes));
            }
            Element node = (Element) nodeElements.item(i);
            String id = node.getAttribute("id");
            double lat = Double.parseDouble(node.getAttribute("lat"));
            double lon = Double.parseDouble(node.getAttribute("lon"));
            nodes.put(id, geometryFactory.createPoint(new Coordinate(lon, lat)));
        }

        // Extract ways
        LOGGER.info("Extracting ways from response...");
        List<LineString> ways = new ArrayList<>();
        NodeList wayElements = doc.getElementsByTagName("way");
        int totalWays = wayElements.getLength();
        LOGGER.info("Found " + totalWays + " ways");
        
        for (int i = 0; i < wayElements.getLength(); i++) {
            if (i % 100 == 0) {
                LOGGER.info(String.format("Processing ways: %.1f%% complete", (i * 100.0) / totalWays));
            }
            Element way = (Element) wayElements.item(i);
            NodeList ndRefs = way.getElementsByTagName("nd");
            List<Coordinate> coordinates = new ArrayList<>();
            
            for (int j = 0; j < ndRefs.getLength(); j++) {
                Element nd = (Element) ndRefs.item(j);
                String ref = nd.getAttribute("ref");
                Point point = nodes.get(ref);
                if (point != null) {
                    coordinates.add(point.getCoordinate());
                }
            }

            if (coordinates.size() >= 2) {
                ways.add(geometryFactory.createLineString(
                    coordinates.toArray(new Coordinate[0])
                ));
            }
        }

        LOGGER.info("Road network download complete. Created " + ways.size() + " valid ways from " + nodes.size() + " nodes");
        Map<String, Object> result = new HashMap<>();
        result.put("nodes", nodes);
        result.put("ways", ways);
        return result;
    }

    public static double calculateLength(LineString line) {
        double length = 0;
        Coordinate[] coords = line.getCoordinates();
        for (int i = 0; i < coords.length - 1; i++) {
            length += calculateDistance(coords[i], coords[i + 1]);
        }
        return length;
    }

    private static double calculateDistance(Coordinate c1, Coordinate c2) {
        final int R = 6371; // Earth's radius in kilometers

        double lat1 = Math.toRadians(c1.y);
        double lat2 = Math.toRadians(c2.y);
        double dLat = Math.toRadians(c2.y - c1.y);
        double dLon = Math.toRadians(c2.x - c1.x);

        double a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                Math.cos(lat1) * Math.cos(lat2) *
                Math.sin(dLon/2) * Math.sin(dLon/2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));

        return R * c;
    }
} 