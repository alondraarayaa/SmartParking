const mongoose = require('mongoose');
const { Schema } = mongoose;

const ParkingSchema = new Schema({
    location: {
        type: String,
        required: true
    },
    occupied: {
        type: Number,
        default: 0
    },
    available: {
        type: Number,
        default: 0
    },
    pricePerHour: {
        type: Number,
        required: true
    },
    createdAt: {
        type: Date,
        default: Date.now
    },
    updatedAt: {
        type: Date,
        default: Date.now
    },
    currentUsers: {
        userId: {
            type: Schema.Types.ObjectId,
            ref: 'User'
        },
        name: {
            type: String
        },
        startTime: {
            type: Date
        },
        endTime: {
            type: Date
        },
        
    }
});


const Parking = mongoose.model('Parking', ParkingSchema);

module.exports = Parking;
