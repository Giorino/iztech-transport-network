package com.izmir.transportation.persistence;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.LineString;

import java.io.IOException;

/**
 * Jackson serializer for JTS LineString geometry objects.
 */
public class LineStringSerializer extends JsonSerializer<LineString> {
    @Override
    public void serialize(LineString lineString, JsonGenerator gen, SerializerProvider provider) throws IOException {
        gen.writeStartObject();
        gen.writeArrayFieldStart("coordinates");
        for (Coordinate coordinate : lineString.getCoordinates()) {
            gen.writeStartArray();
            gen.writeNumber(coordinate.x);
            gen.writeNumber(coordinate.y);
            gen.writeEndArray();
        }
        gen.writeEndArray();
        gen.writeEndObject();
    }
} 